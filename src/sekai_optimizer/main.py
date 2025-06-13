"""
Main entry point for the Sekai Optimizer.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from .workflows.graph_builder import build_optimization_graph
from .workflows.state_initialization import initialize_optimization_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Silence overly verbose library loggers
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def log_iteration_scores(evaluation_history: List[Dict[str, Any]]) -> None:
    """
    Log a comprehensive summary of all iteration cycle scores.

    Args:
        evaluation_history: List of evaluation results from each iteration
    """
    if not evaluation_history:
        logger.info("📊 ITERATION HISTORY: No iterations completed")
        return

    logger.info("=" * 60)
    logger.info("📊 ITERATION CYCLE SCORES SUMMARY")
    logger.info("=" * 60)

    # Header
    logger.info(
        f"{'Iteration':<10} {'P@10 Score':<12} {'Semantic Score':<15} {'Batch Size':<12} {'Status':<10}"
    )
    logger.info("-" * 70)

    # First pass: find the best iteration
    best_p10_score = -1.0
    best_semantic_score = -1.0
    best_iteration = 0

    for i, eval_result in enumerate(evaluation_history, 1):
        p10_score = eval_result.get("average_p10", 0.0)
        semantic_score = eval_result.get("average_semantic_similarity", 0.0) or 0.0

        # Compare: first by P@10, then by semantic similarity if P@10 is equal
        is_better = p10_score > best_p10_score or (
            p10_score == best_p10_score and semantic_score >= best_semantic_score
        )

        if is_better:
            best_p10_score = p10_score
            best_semantic_score = semantic_score
            best_iteration = i

    # Second pass: log all iterations with correct status
    for i, eval_result in enumerate(evaluation_history, 1):
        p10_score = eval_result.get("average_p10", 0.0)
        semantic_score = eval_result.get("average_semantic_similarity", None)
        batch_size = eval_result.get("batch_size", 0)

        # Format semantic score
        semantic_str = f"{semantic_score:.3f}" if semantic_score is not None else "N/A"

        # Only mark the actual best iteration
        status = "🏆 BEST" if i == best_iteration else ""

        logger.info(
            f"#{i:<9} {p10_score:<12.3f} {semantic_str:<15} {batch_size:<12} {status:<10}"
        )

    logger.info("-" * 70)
    logger.info(f"📈 Total Iterations: {len(evaluation_history)}")
    logger.info(
        f"🎯 Best Performance: Iteration #{best_iteration} with P@10 = {best_p10_score:.3f}"
    )

    # Show score progression
    scores = [eval_result.get("average_p10", 0.0) for eval_result in evaluation_history]
    if len(scores) > 1:
        score_changes = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        improvements = sum(1 for change in score_changes if change > 0)
        logger.info(
            f"📊 Score Improvements: {improvements}/{len(score_changes)} iterations"
        )

        if score_changes:
            avg_change = sum(score_changes) / len(score_changes)
            logger.info(f"📈 Average Score Change: {avg_change:+.3f} per iteration")

    logger.info("=" * 60)


def main() -> None:
    """
    Main execution function for the Sekai Optimizer.
    """
    logger.info("=" * 60)
    logger.info("SEKAI OPTIMIZER - STAGE 1 SKELETON")
    logger.info("=" * 60)

    # Load environment variables
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded environment from: {env_path}")
    else:
        logger.warning(f"No .env file found at: {env_path}")

    try:
        # Step 1: Initialize state
        logger.debug("Step 1: Initializing optimization state...")
        initial_state = initialize_optimization_state()

        # Step 2: Build graph
        logger.debug("Step 2: Building optimization workflow graph...")
        graph = build_optimization_graph()

        # Step 3: Execute workflow
        logger.debug("Step 3: Executing optimization workflow...")
        logger.info("=" * 40)
        logger.info("WORKFLOW EXECUTION STARTED")
        logger.info("=" * 40)

        final_state = graph.invoke(
            initial_state,
            {"recursion_limit": int(os.environ.get("MAX_ITERATIONS", 3)) * 10},
        )

        logger.info("=" * 40)
        logger.info("WORKFLOW EXECUTION COMPLETED")
        logger.info("=" * 40)

        # Step 4: Display detailed iteration history
        log_iteration_scores(final_state["evaluation_history"])

        # Step 5: Display optimal results summary
        logger.debug("Step 5: Optimization Results Summary")
        logger.info("🏆 FINAL OPTIMIZATION RESULTS 🏆")
        logger.info(f"✅ Iterations Completed: {final_state['iteration_count']}")
        logger.info(f"🎯 Best Score Achieved: {final_state['best_score']}")
        logger.info(f"📝 Optimal Strategy Prompt:")
        logger.info(f"   {final_state['best_strategy_prompt']}")

        logger.info("=" * 60)
        logger.info("SEKAI OPTIMIZER COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
