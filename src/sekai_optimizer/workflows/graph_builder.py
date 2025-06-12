"""
Graph builder for the Sekai Optimizer workflow.
"""

import logging
import time
import json
from typing import Literal

from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph

from ..data.state import OptimizationState
from ..nodes.pick_users import pick_users_node
from ..nodes.simulate_tags import simulate_tags_node
from ..nodes.recommend_stories import recommend_stories_node
from ..nodes.generate_groundtruths import generate_groundtruths_node
from ..nodes.evaluate import evaluate_node
from ..nodes.optimize_prompt import optimize_prompt_node

logger = logging.getLogger(__name__)


def determine_stop(state: OptimizationState) -> Literal["continue", "end"]:
    """
    Conditional edge to determine whether to continue optimization or stop.

    Stopping conditions:
    1. Max iterations reached
    2. Max time exceeded
    """
    logger.info("=== DETERMINE STOP ===")
    logger.debug(f"Current iteration: {state['iteration_count']}")
    logger.debug(f"Max iterations: {state['config']['max_iterations']}")

    elapsed_minutes = (time.time() - state["start_time"]) / 60.0
    max_minutes = state["config"]["max_optimization_minute"]

    logger.debug(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    logger.debug(f"Max time: {max_minutes} minutes")

    # Check stopping conditions
    if state["iteration_count"] >= state["config"]["max_iterations"]:
        logger.info("STOPPING: Max iterations reached")
        _log_final_results(state)
        return "end"

    if elapsed_minutes >= max_minutes:
        logger.info("STOPPING: Max time exceeded")
        _log_final_results(state)
        return "end"

    logger.debug("CONTINUING: Within iteration and time limits")
    return "continue"


def _log_final_results(state: OptimizationState) -> None:
    """Log the final optimization results."""
    logger.info("=" * 60)
    logger.info("🎯 OPTIMIZATION COMPLETE - FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total iterations completed: {state['iteration_count']}")
    logger.info(f"Best score achieved: {state['best_score']}")
    logger.info(f"Best strategy prompt: {state['best_strategy_prompt']}")
    logger.debug(
        f"Best evaluation details: {json.dumps(state['best_evaluation'], indent=4)}"
    )
    logger.info("=" * 60)


def build_optimization_graph() -> CompiledGraph:
    """
    Build the complete optimization workflow graph.

    Returns:
        Compiled LangGraph ready for execution
    """
    logger.debug("Building optimization workflow graph...")

    # Create the graph
    workflow = StateGraph(OptimizationState)

    # Add nodes
    workflow.add_node("pick_users", pick_users_node)
    workflow.add_node("simulate_tags", simulate_tags_node)
    workflow.add_node("recommend_stories", recommend_stories_node)
    workflow.add_node("generate_groundtruths", generate_groundtruths_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("optimize_prompt", optimize_prompt_node)

    # Set entry point
    workflow.set_entry_point("pick_users")

    # Add sequential edges
    workflow.add_edge("pick_users", "simulate_tags")
    workflow.add_edge("simulate_tags", "recommend_stories")
    workflow.add_edge("recommend_stories", "generate_groundtruths")
    workflow.add_edge("generate_groundtruths", "evaluate")
    workflow.add_edge("evaluate", "optimize_prompt")

    # Add conditional edge (optimization loop)
    workflow.add_conditional_edges(
        "optimize_prompt",
        determine_stop,
        {
            "continue": "pick_users",  # Loop back to start new iteration
            "end": "__end__",  # Terminate the workflow
        },
    )

    # Compile the graph
    graph = workflow.compile()

    logger.debug("Optimization workflow graph built successfully")
    return graph
