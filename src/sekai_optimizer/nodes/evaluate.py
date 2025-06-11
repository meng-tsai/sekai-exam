"""
Evaluate node implementation.
"""

import logging
from typing import Dict, Any, List
import time

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


def evaluate_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Produce a single, generalizable evaluation report for the entire batch.

    State I/O Contract:
    - Reads from State: state['batch_recommendations'], state['batch_ground_truths'], state['current_user_batch']
    - Updates State with: { "evaluation_result": Dict }
    """
    logger.info("=== EVALUATE NODE ===")
    logger.info(f"Evaluating batch of {len(state['current_user_batch'])} users")
    logger.info(f"Recommendations count: {len(state['batch_recommendations'])}")
    logger.info(f"Ground truths count: {len(state['batch_ground_truths'])}")

    # STUB: Generate mock evaluation metrics
    individual_scores = []

    for user in state["current_user_batch"]:
        user_id = user.get("user_id", "unknown")

        if (
            user_id in state["batch_recommendations"]
            and user_id in state["batch_ground_truths"]
        ):
            # Mock: Calculate overlap (intersection) between recommendations and ground truth
            recommendations = set(state["batch_recommendations"][user_id])
            ground_truth = set(state["batch_ground_truths"][user_id])
            overlap = len(recommendations.intersection(ground_truth))
            precision_at_10 = overlap / 10.0  # P@10 score
            # TODO: Add semantic similarity score
            individual_scores.append(precision_at_10)

            logger.info(
                f"User {user_id}: P@10 = {precision_at_10:.3f} ({overlap}/10 overlap)"
            )

    # Mock aggregate metrics
    average_p10 = (
        sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
    )
    mock_synthesized_feedback = f"Mock feedback: Average P@10 is {average_p10:.3f}. Consider improving tag matching."

    # TODO: Add semantic similarity score
    mock_evaluation_result = {
        "average_p10": average_p10,
        "individual_scores": individual_scores,
        "synthesized_feedback": mock_synthesized_feedback,
        "batch_size": len(state["current_user_batch"]),
    }

    logger.info(f"Batch evaluation complete: Avg P@10 = {average_p10:.3f}")
    logger.info(f"Synthesized feedback: {mock_synthesized_feedback}")
    time.sleep(10)
    return {"evaluation_result": mock_evaluation_result}
