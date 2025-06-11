"""
Pick users node implementation.
"""

import logging
from typing import Dict, Any

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


def pick_users_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Select a random batch of users from the training set.

    State I/O Contract:
    - Reads from State: state['full_training_users'], state['config']['user_per_batch']
    - Updates State with: { "current_user_batch": List[Dict] }
    """
    logger.info("=== PICK USERS NODE ===")
    logger.info(f"Total training users available: {len(state['full_training_users'])}")
    logger.info(f"Batch size requested: {state['config']['user_per_batch']}")

    # STUB: Return first N users for now
    batch_size = min(
        state["config"]["user_per_batch"], len(state["full_training_users"])
    )
    mock_batch = state["full_training_users"][:batch_size]

    logger.info(f"Selected {len(mock_batch)} users for current batch")
    logger.info(
        f"Selected user IDs: {[user.get('user_id', 'unknown') for user in mock_batch]}"
    )

    return {"current_user_batch": mock_batch}
