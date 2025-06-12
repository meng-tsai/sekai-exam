"""
Evaluate node implementation with hybrid scoring and LLM feedback synthesis.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langsmith import Client as LangSmithClient
from pydantic import BaseModel

from ..data.state import OptimizationState
from .recommend_stories import RecommendationService

logger = logging.getLogger(__name__)


class FeedbackSynthesisResponse(BaseModel):
    """Pydantic model for the expected LLM feedback synthesis response format."""

    feedback: str


class EvaluationService:
    """Singleton service for evaluation metrics and feedback synthesis."""

    _instance = None

    def __init__(self):
        """Initialize the evaluation service."""
        self.openai_client = openai.OpenAI()
        self.story_service = RecommendationService.get_instance()

    @classmethod
    def get_instance(cls):
        """Get singleton instance of EvaluationService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def calculate_precision_at_10(
        self, recommended: List[int], ground_truth: List[int]
    ) -> float:
        """Calculate Precision@10 score between recommended and ground truth stories."""
        recommended_set = set(recommended)
        ground_truth_set = set(ground_truth)
        overlap = len(recommended_set.intersection(ground_truth_set))
        return overlap / 10.0  # P@10 with denominator of 10

    def _format_story_for_embedding(self, story: Dict) -> str:
        """Format story for embedding using consistent format."""
        return f"{story['title']}: {story.get('intro', '')}"

    def _embed_story_set(self, stories: List[Dict]) -> np.ndarray:
        """Embed a set of stories as concatenated text."""
        try:
            # Concatenate all stories with separator
            story_texts = [self._format_story_for_embedding(story) for story in stories]
            concatenated_text = " | ".join(story_texts)

            # Get embedding
            response = self.openai_client.embeddings.create(
                input=[concatenated_text], model="text-embedding-3-small"
            )
            embedding = np.array(response.data[0].embedding, dtype="f4")
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed story set: {e}")
            raise

    def calculate_semantic_similarity(
        self, recommended_stories: List[Dict], ground_truth_stories: List[Dict]
    ) -> Optional[float]:
        """
        Calculate semantic similarity between recommended and ground truth story sets.

        Returns None if embedding calculation fails.
        """
        try:
            # Embed both story sets
            recommended_embedding = self._embed_story_set(recommended_stories)
            ground_truth_embedding = self._embed_story_set(ground_truth_stories)

            # Calculate cosine similarity
            dot_product = np.dot(recommended_embedding, ground_truth_embedding)
            norm_recommended = np.linalg.norm(recommended_embedding)
            norm_ground_truth = np.linalg.norm(ground_truth_embedding)

            if norm_recommended == 0 or norm_ground_truth == 0:
                return 0.0

            similarity = dot_product / (norm_recommended * norm_ground_truth)
            return float(similarity)

        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return None

    def evaluate_single_user(
        self, user_data: Dict, recommendations: List[int], ground_truths: List[int]
    ) -> Dict:
        """Evaluate a single user's recommendations."""
        user_id = str(user_data.get("user_id", "unknown"))

        # Get story data for both sets
        recommended_stories = []
        ground_truth_stories = []

        for story_id in recommendations:
            if story_id in self.story_service.story_data:
                recommended_stories.append(self.story_service.story_data[story_id])

        for story_id in ground_truths:
            if story_id in self.story_service.story_data:
                ground_truth_stories.append(self.story_service.story_data[story_id])

        # Calculate P@10
        precision_at_10 = self.calculate_precision_at_10(recommendations, ground_truths)

        # Calculate semantic similarity
        semantic_similarity = None
        if recommended_stories and ground_truth_stories:
            semantic_similarity = self.calculate_semantic_similarity(
                recommended_stories, ground_truth_stories
            )

        # Count overlap for logging
        overlap_count = len(set(recommendations).intersection(set(ground_truths)))

        result = {
            "user_id": user_id,
            "precision_at_10": precision_at_10,
            "semantic_similarity": semantic_similarity,
            "overlap_count": overlap_count,
            "recommended_stories": recommended_stories,
            "ground_truth_stories": ground_truth_stories,
        }

        semantic_sim_str = (
            f"{semantic_similarity:.3f}"
            if semantic_similarity is not None
            else "Failed"
        )
        logger.info(
            f"User {user_id}: P@10={precision_at_10:.3f}, "
            f"Semantic={semantic_sim_str}, "
            f"Overlap={overlap_count}/10"
        )

        return result

    def evaluate_user_batch(
        self,
        users: List[Dict],
        batch_recommendations: Dict[str, List[int]],
        batch_ground_truths: Dict[str, List[int]],
    ) -> Dict[str, Dict]:
        """Evaluate a batch of users using RunnableParallel."""
        try:
            # Create individual evaluation functions for each user
            user_runnables = {}
            for user in users:
                user_id = str(user.get("user_id", "unknown"))
                if user_id in batch_recommendations and user_id in batch_ground_truths:
                    user_runnables[user_id] = RunnableLambda(
                        lambda _, uid=user_id, user_data=user: self.evaluate_single_user(
                            user_data,
                            batch_recommendations[uid],
                            batch_ground_truths[uid],
                        )
                    )

            # Execute in parallel
            if user_runnables:
                parallel_runnable = RunnableParallel(user_runnables)
                results = parallel_runnable.invoke({})
                logger.info(f"Evaluated {len(results)} users in parallel")
                return results
            else:
                logger.warning("No valid users found for evaluation")
                return {}

        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise

    def synthesize_feedback(
        self,
        current_strategy_prompt: str,
        individual_results: Dict[str, Dict],
        average_p10: float,
        average_semantic_similarity: Optional[float],
        batch_size: int,
    ) -> str:
        """Synthesize actionable feedback using LangSmith."""
        try:
            # Setup LangSmith client and chain
            langsmith_client = LangSmithClient(
                api_key=os.environ.get("LANGSMITH_API_KEY")
            )

            # Pull prompt with model from LangSmith Hub
            feedback_synthesizer_runnable = langsmith_client.pull_prompt(
                "evaluate_feedback_synthesizer:latest", include_model=True
            )

            # Setup JSON parser
            json_parser = JsonOutputParser(pydantic_object=FeedbackSynthesisResponse)
            chain = feedback_synthesizer_runnable | json_parser

            # Format batch evaluation data for prompt
            batch_evaluation_data = []
            for user_id, result in individual_results.items():
                user_summary = {
                    "user_id": user_id,
                    "precision_at_10": result["precision_at_10"],
                    "semantic_similarity": result["semantic_similarity"],
                    "overlap_count": result["overlap_count"],
                    "recommended_stories": [
                        {
                            "id": story.get("id", "unknown"),
                            "title": story.get("title", "Unknown Title"),
                            "intro": story.get("intro", ""),
                            "tags": story.get("tags", []),
                        }
                        for story in result["recommended_stories"]
                    ],
                    "ground_truth_stories": [
                        {
                            "id": story.get("id", "unknown"),
                            "title": story.get("title", "Unknown Title"),
                            "intro": story.get("intro", ""),
                            "tags": story.get("tags", []),
                        }
                        for story in result["ground_truth_stories"]
                    ],
                }
                batch_evaluation_data.append(user_summary)

            # Call LLM for feedback synthesis
            feedback_result = chain.invoke(
                {
                    "current_strategy_prompt": current_strategy_prompt,
                    "batch_evaluation_data": json.dumps(
                        batch_evaluation_data, indent=2
                    ),
                    "average_p10": f"{average_p10:.3f}",
                    "average_semantic_similarity": (
                        f"{average_semantic_similarity:.3f}"
                        if average_semantic_similarity is not None
                        else "N/A (calculation failed)"
                    ),
                    "batch_size": batch_size,
                }
            )

            synthesized_feedback = feedback_result.get(
                "feedback", "No feedback generated"
            )
            logger.info(f"Generated synthesized feedback.")
            return synthesized_feedback

        except Exception as e:
            logger.error(f"Feedback synthesis failed: {e}")
            # Hard failure as specified
            raise


def evaluate_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Produce a single, generalizable evaluation report for the entire batch.

    State I/O Contract:
    - Reads from State: state['batch_recommendations'], state['batch_ground_truths'], state['current_user_batch']
    - Updates State with: { "evaluation_result": Dict }
    """
    logger.info("=== EVALUATE NODE ===")
    logger.info(f"Evaluating batch of {len(state['current_user_batch'])} users")

    users = state["current_user_batch"]
    batch_recommendations = state["batch_recommendations"]
    batch_ground_truths = state["batch_ground_truths"]
    current_strategy_prompt = state["current_strategy_prompt"]

    try:
        # Get evaluation service instance
        service = EvaluationService.get_instance()

        # Evaluate all users in batch
        individual_results = service.evaluate_user_batch(
            users, batch_recommendations, batch_ground_truths
        )

        # Calculate aggregate metrics
        precision_scores = [
            result["precision_at_10"] for result in individual_results.values()
        ]
        semantic_scores = [
            result["semantic_similarity"]
            for result in individual_results.values()
            if result["semantic_similarity"] is not None
        ]

        average_p10 = (
            sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        )
        average_semantic_similarity = (
            sum(semantic_scores) / len(semantic_scores) if semantic_scores else None
        )

        failed_semantic_calculations = len(individual_results) - len(semantic_scores)

        # Synthesize feedback using LLM
        synthesized_feedback = service.synthesize_feedback(
            current_strategy_prompt,
            individual_results,
            average_p10,
            average_semantic_similarity,
            len(users),
        )

        # Prepare final evaluation result
        evaluation_result = {
            "average_p10": average_p10,
            "average_semantic_similarity": average_semantic_similarity,
            "synthesized_feedback": synthesized_feedback,
            "individual_scores": list(individual_results.values()),
            "batch_size": len(users),
            "failed_semantic_calculations": failed_semantic_calculations,
        }

        logger.info(f"Batch evaluation complete:")
        logger.info(f"  Average P@10: {average_p10:.3f}")
        semantic_sim_str = (
            f"{average_semantic_similarity:.3f}"
            if average_semantic_similarity is not None
            else "N/A"
        )
        logger.info(f"  Average Semantic Similarity: {semantic_sim_str}")
        logger.info(f"  Failed Semantic Calculations: {failed_semantic_calculations}")
        logger.info(f"  Synthesized Feedback:\n{synthesized_feedback}")

        return {"evaluation_result": evaluation_result}

    except Exception as e:
        logger.error(f"Critical error in evaluation: {e}", exc_info=True)
        # Hard failure as specified
        raise
