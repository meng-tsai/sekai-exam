"""
Pick users node implementation.
"""

import logging
import random
import time
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
    logger.debug(f"Total training users available: {len(state['full_training_users'])}")
    logger.debug(f"Batch size requested: {state['config']['user_per_batch']}")

    # Use current time as random seed for completely random selection
    random.seed(time.time())

    # Calculate actual batch size (can't exceed available users)
    batch_size = min(
        state["config"]["user_per_batch"], len(state["full_training_users"])
    )

    # Random selection without replacement
    if batch_size == 0 or len(state["full_training_users"]) == 0:
        selected_batch = []
    elif batch_size >= len(state["full_training_users"]):
        # Return all users if batch size >= total users
        selected_batch = state["full_training_users"].copy()
    else:
        # Use random.sample for proper random selection without replacement
        selected_batch = random.sample(state["full_training_users"], batch_size)

    logger.debug(f"Selected {len(selected_batch)} users for current batch")
    logger.debug(
        f"Selected user IDs: {[user.get('user_id', 'unknown') for user in selected_batch]}"
    )

    return {"current_user_batch": selected_batch}
