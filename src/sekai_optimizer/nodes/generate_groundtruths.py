"""
Generate ground truths node implementation.
"""

import logging
from typing import Dict, Any, List

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


def generate_groundtruths_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Generate the "gold-standard" recommendations for each user.

    State I/O Contract:
    - Reads from State: state['current_user_batch']
    - Updates State with: { "batch_ground_truths": Dict[str, List[int]] }
    """
    logger.info("=== GENERATE GROUND TRUTHS NODE ===")
    logger.info(
        f"Generating ground truths for {len(state['current_user_batch'])} users"
    )
    logger.info(f"Total stories available: {len(state['all_stories'])}")

    # STUB: Generate mock ground truth recommendations for each user
    mock_ground_truths: Dict[str, List[int]] = {}

    for user in state["current_user_batch"]:
        user_id = user.get("user_id", "unknown")
        # Mock: Return last 10 story IDs as "ground truth" (different from recommendations)
        story_ids = [story["id"] for story in state["all_stories"][-10:]]
        mock_ground_truths[user_id] = story_ids

        logger.info(f"User {user_id}: ground truth stories = {story_ids}")

    logger.info(
        f"Completed ground truth generation for {len(mock_ground_truths)} users"
    )

    return {"batch_ground_truths": mock_ground_truths}
