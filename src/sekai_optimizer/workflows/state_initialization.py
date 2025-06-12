"""
State initialization utilities for the optimization workflow.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


def load_training_data() -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """
    Load training users and stories from data files.

    Returns:
        Tuple of (users, stories) loaded from JSON files
    """
    data_dir = Path(__file__).parent.parent / "data"

    # Load users
    users_file = data_dir / "users.json"
    if not users_file.exists():
        logger.error(f"Users file not found: {users_file}")
        raise FileNotFoundError(f"Users data file not found: {users_file}")

    with open(users_file, "r") as f:
        users = json.load(f)
    logger.debug(f"Loaded {len(users)} training users")

    # Load stories
    stories_file = data_dir / "stories.json"
    if not stories_file.exists():
        logger.error(f"Stories file not found: {stories_file}")
        raise FileNotFoundError(f"Stories data file not found: {stories_file}")

    with open(stories_file, "r") as f:
        stories = json.load(f)
    logger.debug(f"Loaded {len(stories)} stories")

    return users, stories


def create_initial_config() -> Dict[str, Any]:
    """
    Create configuration dictionary from environment variables.

    Returns:
        Configuration dictionary with all required settings
    """
    config = {
        "user_per_batch": int(os.getenv("USER_PER_BATCH", "3")),
        "max_iterations": int(os.getenv("MAX_ITERATIONS", "3")),
        "max_optimization_minute": int(os.getenv("MAX_OPTIMIZATION_MINUTE", "5")),
        "init_strategy_prompt": os.getenv(
            "INIT_STRATEGY_PROMPT",
            "Recommend stories that match user preferences and tags",
        ),
    }

    logger.debug("Configuration loaded:")
    for key, value in config.items():
        logger.debug(f"  {key}: {value}")

    return config


def get_faiss_index_path() -> str:
    """
    Get the path to the FAISS index file.

    Returns:
        Path to the FAISS index file
    """
    data_dir = Path(__file__).parent.parent / "data"
    index_path = str(data_dir / "stories.index")

    if not Path(index_path).exists():
        logger.warning(f"FAISS index file not found: {index_path}")
        logger.warning("This is expected in Stage 1 - using stub path")

    return index_path


def initialize_optimization_state() -> OptimizationState:
    """
    Initialize the complete optimization state.

    Returns:
        Fully initialized OptimizationState ready for workflow execution
    """
    logger.info("Initializing optimization state...")

    # Load training data
    users, stories = load_training_data()

    # Create configuration
    config = create_initial_config()

    # Get FAISS index path
    faiss_index_path = get_faiss_index_path()

    # Create initial state
    initial_state: OptimizationState = {
        # User-related properties
        "full_training_users": users,
        "current_user_batch": [],
        # Strategy properties
        "current_strategy_prompt": config["init_strategy_prompt"],
        "iteration_count": 0,
        # Results properties
        "batch_simulated_tags": {},
        "batch_recommendations": {},
        "batch_ground_truths": {},
        "evaluation_result": {},
        "evaluation_history": [],
        # Best result tracking
        "best_strategy_prompt": config["init_strategy_prompt"],
        "best_score": -1.0,
        "best_evaluation": {},
        # Control properties
        "start_time": time.time(),
        "config": config,
        # Data resources
        "all_stories": stories,
        "faiss_index_path": faiss_index_path,
    }

    logger.debug("Optimization state initialized successfully")
    logger.debug(f"Training users: {len(users)}")
    logger.debug(f"Available stories: {len(stories)}")
    logger.debug(f"Initial strategy prompt: {config['init_strategy_prompt'][:50]}...")

    return initial_state
