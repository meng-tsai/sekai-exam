"""
Optimize prompt node implementation.
"""

import logging
from typing import Dict, Any, List

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


def optimize_prompt_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Generate the next-generation strategy prompt.

    State I/O Contract:
    - Reads from State: state['evaluation_result'], state['current_strategy_prompt'], state['evaluation_history']
    - Updates State with: { "current_strategy_prompt": str, "evaluation_history": List[Dict], "iteration_count": int,
                          "best_strategy_prompt": str, "best_score": float, "best_evaluation": Dict }
    """
    logger.info("=== OPTIMIZE PROMPT NODE ===")
    logger.info(f"Current iteration: {state['iteration_count']}")
    logger.info(f"Current strategy prompt: {state['current_strategy_prompt'][:100]}...")
    logger.info(f"Evaluation result: {state['evaluation_result']}")
    logger.info(f"Historical evaluations: {len(state['evaluation_history'])}")

    # Get current score
    current_score = state["evaluation_result"].get("average_p10", 0.0)
    current_prompt = state["current_strategy_prompt"]

    # Check if this is the best score so far
    is_best = current_score > state["best_score"]
    best_prompt = state["best_strategy_prompt"]
    best_score = state["best_score"]
    best_evaluation = state["best_evaluation"]

    if is_best:
        logger.info(f"🎉 NEW BEST SCORE! {current_score} > {state['best_score']}")
        best_prompt = current_prompt
        best_score = current_score
        best_evaluation = state["evaluation_result"].copy()
    else:
        logger.info(
            f"Score {current_score} did not beat best score {state['best_score']}"
        )

    logger.info(f"Current best prompt: {best_prompt[:100]}...")
    logger.info(f"Current best score: {best_score}")

    # STUB: Generate next strategy prompt (just append iteration info)
    next_iteration = state["iteration_count"] + 1
    feedback = state["evaluation_result"].get("synthesized_feedback", "No feedback")
    new_strategy_prompt = (
        f"ITERATION {next_iteration}: {current_prompt} [FEEDBACK: {feedback[:50]}...]"
    )

    logger.info(f"Generated new strategy prompt: {new_strategy_prompt[:100]}...")

    # Update evaluation history
    updated_history = state["evaluation_history"] + [state["evaluation_result"]]
    logger.info(f"Updated evaluation history (now {len(updated_history)} entries)")

    return {
        "current_strategy_prompt": new_strategy_prompt,
        "evaluation_history": updated_history,
        "iteration_count": next_iteration,
        "best_strategy_prompt": best_prompt,
        "best_score": best_score,
        "best_evaluation": best_evaluation,
    }
