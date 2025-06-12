"""
Main entry point for the Sekai Optimizer.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .workflows.graph_builder import build_optimization_graph
from .workflows.state_initialization import initialize_optimization_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


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
        logger.info(f"Loaded environment from: {env_path}")
    else:
        logger.warning(f"No .env file found at: {env_path}")

    try:
        # Step 1: Initialize state
        logger.info("Step 1: Initializing optimization state...")
        initial_state = initialize_optimization_state()

        # Step 2: Build graph
        logger.info("Step 2: Building optimization workflow graph...")
        graph = build_optimization_graph()

        # Step 3: Execute workflow
        logger.info("Step 3: Executing optimization workflow...")
        logger.info("=" * 40)
        logger.info("WORKFLOW EXECUTION STARTED")
        logger.info("=" * 40)

        final_state = graph.invoke(initial_state)

        logger.info("=" * 40)
        logger.info("WORKFLOW EXECUTION COMPLETED")
        logger.info("=" * 40)

        # Step 4: Display optimal results
        logger.info("Step 4: Optimization Results Summary")
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
