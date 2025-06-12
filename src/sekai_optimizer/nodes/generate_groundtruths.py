"""
Generate ground truths node implementation.
"""

import logging
import os
from typing import Dict, Any, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langsmith import Client as LangSmithClient
from pydantic import BaseModel

from ..data.state import OptimizationState
from .recommend_stories import RecommendationService

logger = logging.getLogger(__name__)


class GroundTruthResponse(BaseModel):
    """Pydantic model for the expected LLM response format."""

    recommendations: List[int]


def _generate_ground_truth_for_user(
    user_profile: str, complete_stories_formatted: str, service: RecommendationService
) -> List[int]:
    """Generate ground truth for a single user."""
    try:
        # Setup LangSmith client and chain
        langsmith_client = LangSmithClient(api_key=os.environ.get("LANGSMITH_API_KEY"))

        # Pull prompt with model from LangSmith Hub
        groundtruth_runnable = langsmith_client.pull_prompt(
            "generate_groundtruths:latest", include_model=True
        )

        # Setup JSON parser with Pydantic model
        json_parser = JsonOutputParser(pydantic_object=GroundTruthResponse)
        chain = groundtruth_runnable | json_parser

        # Call LLM for ground truth generation
        result = chain.invoke(
            {
                "user_profile": user_profile,
                "complete_stories": complete_stories_formatted,
            }
        )

        # Extract and validate recommendations
        recommendations = result.get("recommendations", [])

        # Validate that we have exactly 10 recommendations and they're valid story IDs
        valid_recommendations = [
            rec
            for rec in recommendations
            if isinstance(rec, int) and rec in service.story_data
        ]

        # Ensure exactly 10 recommendations
        if len(valid_recommendations) < 10:
            logger.warning(
                f"Only {len(valid_recommendations)} valid recommendations, "
                f"padding with additional stories"
            )
            # Pad with remaining stories if needed
            all_story_ids = list(service.story_data.keys())
            for story_id in all_story_ids:
                if story_id not in valid_recommendations:
                    valid_recommendations.append(story_id)
                    if len(valid_recommendations) >= 10:
                        break

        final_recommendations = valid_recommendations[:10]
        return final_recommendations

    except Exception as e:
        logger.error(f"Failed to generate ground truth: {e}")
        # Hard fail as requested (Option C)
        raise


def generate_groundtruths_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Generate gold-standard recommendations for each user using complete story dataset and full user profiles.

    State I/O Contract:
    - Reads from State: state['current_user_batch']
    - Updates State with: { "batch_ground_truths": Dict[str, List[int]] }
    """
    logger.info("=== GENERATE GROUND TRUTHS NODE ===")

    users = state["current_user_batch"]
    logger.info(f"Generating ground truths for {len(users)} users")

    try:
        # Get recommendation service instance for story data access
        service = RecommendationService.get_instance()

        # Format complete story dataset for prompt (done once)
        complete_stories_text = []
        for story_id, story in service.story_data.items():
            story_text = f"ID: {story_id}, Title: {story['title']}, Intro: {story.get('intro', '')}, Tags: {', '.join(story.get('tags', []))}"
            complete_stories_text.append(story_text)

        complete_stories_formatted = "\n".join(complete_stories_text)

        # Create individual ground truth functions for each user
        user_runnables = {}
        for user in users:
            user_id = str(user.get("user_id", "unknown"))
            user_profile = user.get("profile", "")

            user_runnables[user_id] = RunnableLambda(
                lambda _, uid=user_id, profile=user_profile: _generate_ground_truth_for_user(
                    profile, complete_stories_formatted, service
                )
            )

        # Execute in parallel
        parallel_runnable = RunnableParallel(user_runnables)
        batch_ground_truths = parallel_runnable.invoke({})

        # Log results
        for user_id, ground_truths in batch_ground_truths.items():
            logger.info(f"User {user_id}: ground truth = {ground_truths}")

        logger.info(
            f"Completed ground truth generation for {len(batch_ground_truths)} users"
        )

        return {"batch_ground_truths": batch_ground_truths}

    except Exception as e:
        logger.error(f"Critical error in ground truth generation: {e}", exc_info=True)
        # Hard fail the graph as requested
        raise
