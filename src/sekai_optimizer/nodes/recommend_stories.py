"""
Recommend stories node implementation.
"""

import logging
from typing import Dict, Any, List

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


def recommend_stories_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Generate 10 story recommendations for each user.

    State I/O Contract:
    - Reads from State: state['batch_simulated_tags'], state['current_strategy_prompt']
    - Updates State with: { "batch_recommendations": Dict[str, List[int]] }
    """
    logger.info("=== RECOMMEND STORIES NODE ===")
    logger.info(f"Current strategy prompt: {state['current_strategy_prompt'][:100]}...")
    logger.info(
        f"Processing recommendations for {len(state['batch_simulated_tags'])} users"
    )
    logger.info(f"Total stories available: {len(state['all_stories'])}")

    # STUB: Generate mock recommendations for each user
    mock_recommendations: Dict[str, List[int]] = {}

    for user_id, tags in state["batch_simulated_tags"].items():
        # Mock: Return first 10 story IDs for each user
        story_ids = [story["id"] for story in state["all_stories"][:10]]
        mock_recommendations[user_id] = story_ids

        logger.info(
            f"User {user_id} with tags {tags}: recommended stories = {story_ids}"
        )

    logger.info(f"Completed recommendations for {len(mock_recommendations)} users")

    return {"batch_recommendations": mock_recommendations}
