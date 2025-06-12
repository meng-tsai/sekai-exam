"""
Simulate tags node implementation.
"""

import logging
import os
from typing import Dict, Any, List

from langchain_core.output_parsers import JsonOutputParser
from langsmith import Client as LangSmithClient
from pydantic import BaseModel

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


class UserTagsResponse(BaseModel):
    """Pydantic model for the expected LLM response format."""

    user_tags: Dict[str, List[str]]


def simulate_tags_node(state: OptimizationState) -> Dict[str, Any]:
    """
    For each user in the batch, simulate the tags they would select.

    State I/O Contract:
    - Reads from State: state['current_user_batch']
    - Updates State with: { "batch_simulated_tags": Dict[str, List[str]] }
    """
    logger.info("=== SIMULATE TAGS NODE ===")
    logger.info(
        f"Processing {len(state['current_user_batch'])} users for tag simulation"
    )

    users = state["current_user_batch"]

    try:
        # Setup LangSmith client and chain
        langsmith_client = LangSmithClient(api_key=os.environ.get("LANGSMITH_API_KEY"))

        # Pull prompt with model from LangSmith Hub
        tag_simulator_runnable = langsmith_client.pull_prompt(
            "simulate_tags:latest", include_model=True
        )

        # Setup JSON parser with Pydantic model
        json_parser = JsonOutputParser(pydantic_object=UserTagsResponse)
        chain = tag_simulator_runnable | json_parser

        # Prepare batch input for LLM
        user_profiles = [
            {"user_id": str(user.get("user_id", "unknown")), "profile": user["profile"]}
            for user in users
        ]

        logger.info(f"Sending batch of {len(user_profiles)} user profiles to LLM")

        # Single batch LLM call
        result = chain.invoke({"user_batch": user_profiles})

        # Process results with error handling
        batch_simulated_tags: Dict[str, List[str]] = {}

        for user in users:
            user_id = str(user.get("user_id", "unknown"))
            try:
                # Extract tags for this user from LLM response
                tags = result.get("user_tags", {}).get(user_id, [])
                batch_simulated_tags[user_id] = tags
                logger.info(f"User {user_id}: simulated tags = {tags}")
            except Exception as e:
                logger.error(f"Failed to get tags for user {user_id}: {e}")
                # Use empty list for failed users
                batch_simulated_tags[user_id] = []

        logger.info(f"Completed tag simulation for {len(batch_simulated_tags)} users")

        return {"batch_simulated_tags": batch_simulated_tags}

    except Exception as e:
        logger.error(f"Critical error in tag simulation: {e}", exc_info=True)

        # Fallback: return empty tags for all users
        fallback_tags: Dict[str, List[str]] = {
            str(user.get("user_id", "unknown")): [] for user in users
        }

        logger.warning(f"Using fallback empty tags for {len(fallback_tags)} users")
        return {"batch_simulated_tags": fallback_tags}
