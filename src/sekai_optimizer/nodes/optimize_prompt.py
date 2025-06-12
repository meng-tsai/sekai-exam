"""
Optimize prompt node implementation.
"""

import json
import logging
import os
from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langsmith import Client as LangSmithClient
from pydantic import BaseModel

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


class OptimizedPromptResponse(BaseModel):
    """Pydantic model for the expected LLM response format."""

    new_strategy_prompt: str


def optimize_prompt_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Generate the next-generation strategy prompt using LLM-powered optimization.

    State I/O Contract:
    - Reads from State: state['evaluation_result'], state['current_strategy_prompt'], state['evaluation_history']
    - Updates State with: { "current_strategy_prompt": str, "evaluation_history": List[Dict], "iteration_count": int,
                          "best_strategy_prompt": str, "best_score": float, "best_evaluation": Dict }
    """
    logger.info("=== OPTIMIZE PROMPT NODE ===")
    logger.info(f"Current iteration: {state['iteration_count']}")
    logger.info(f"Current strategy prompt: {state['current_strategy_prompt'][:100]}...")
    logger.info(
        "Evaluation result:\n%s", json.dumps(state["evaluation_result"], indent=2)
    )
    logger.info(
        "Historical evaluations: %s", json.dumps(state["evaluation_history"], indent=2)
    )

    # Get current score and evaluation data
    current_score = state["evaluation_result"].get("average_p10", 0.0)
    current_prompt = state["current_strategy_prompt"]
    current_feedback = state["evaluation_result"].get(
        "synthesized_feedback", "No feedback"
    )

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

    try:
        # Setup LangSmith client and chain
        langsmith_client = LangSmithClient(api_key=os.environ.get("LANGSMITH_API_KEY"))

        # Pull prompt with model from LangSmith Hub
        optimizer_runnable = langsmith_client.pull_prompt(
            "optimize_recommendation:latest", include_model=True
        )

        # Setup JSON parser with Pydantic model
        json_parser = JsonOutputParser(pydantic_object=OptimizedPromptResponse)
        chain = optimizer_runnable | json_parser

        # Format evaluation history for prompt
        history_text = ""
        if state["evaluation_history"]:
            history_entries = []
            for i, eval_result in enumerate(state["evaluation_history"], 1):
                score = eval_result.get("average_p10", 0.0)
                feedback = eval_result.get("synthesized_feedback", "No feedback")
                history_entries.append(f"Iteration {i}: Score {score:.3f} - {feedback}")
            history_text = "\n".join(history_entries)
        else:
            history_text = "No previous evaluations"

        logger.info(f"Sending optimization request to LLM")
        logger.info(f"Current score: {current_score}")
        logger.info(f"Current feedback: {current_feedback}")

        # Call LLM for prompt optimization
        result = chain.invoke(
            {
                "current_prompt": current_prompt,
                "current_score": current_score,
                "current_feedback": current_feedback,
                "evaluation_history": history_text,
            }
        )

        # Extract the new strategy prompt
        new_strategy_prompt = result["new_strategy_prompt"]
        logger.info(f"Generated new strategy prompt: {new_strategy_prompt[:100]}...")

    except Exception as e:
        logger.error(f"Critical error in prompt optimization: {e}", exc_info=True)
        # Hard fail as requested
        raise

    # Update evaluation history
    updated_history = state["evaluation_history"] + [state["evaluation_result"]]
    next_iteration = state["iteration_count"] + 1

    logger.info(f"Updated evaluation history (now {len(updated_history)} entries)")
    logger.info(f"Next iteration: {next_iteration}")

    return {
        "current_strategy_prompt": new_strategy_prompt,
        "evaluation_history": updated_history,
        "iteration_count": next_iteration,
        "best_strategy_prompt": best_prompt,
        "best_score": best_score,
        "best_evaluation": best_evaluation,
    }
