"""
State definitions for the Sekai Optimizer workflow.
"""

from typing import TypedDict, List, Dict, Any


class OptimizationState(TypedDict):
    """Complete state for the optimization workflow."""

    # User-related properties
    full_training_users: List[Dict[str, Any]]
    current_user_batch: List[Dict[str, Any]]

    # Strategy properties
    current_strategy_prompt: str
    iteration_count: int

    # Results properties
    batch_simulated_tags: Dict[str, List[str]]  # user_id -> tags
    batch_recommendations: Dict[str, List[int]]  # user_id -> story_ids
    batch_ground_truths: Dict[str, List[int]]  # user_id -> story_ids
    evaluation_result: Dict[str, Any]
    evaluation_history: List[Dict[str, Any]]

    # Best result tracking
    best_strategy_prompt: str
    best_score: float
    best_evaluation: Dict[str, Any]

    # Control properties
    start_time: float
    config: Dict[str, Any]

    # Data resources
    all_stories: List[Dict[str, Any]]
    faiss_index_path: str
