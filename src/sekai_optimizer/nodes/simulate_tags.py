"""
Simulate tags node implementation.
"""

import logging
from typing import Dict, Any, List

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


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

    # STUB: Generate mock tags for each user
    mock_simulated_tags: Dict[str, List[str]] = {}

    for user in state["current_user_batch"]:
        user_id = user.get("user_id", "unknown")
        # Mock tags based on common anime genres
        mock_tags = ["action", "romance", "supernatural", "comedy", "drama"]
        mock_simulated_tags[user_id] = mock_tags

        logger.info(f"User {user_id}: simulated tags = {mock_tags}")

    logger.info(f"Completed tag simulation for {len(mock_simulated_tags)} users")

    return {"batch_simulated_tags": mock_simulated_tags}
