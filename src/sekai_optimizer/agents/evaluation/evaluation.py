"""Main evaluation agent for sekai optimizer."""

import json
import logging
import os
from typing import List, TypedDict, Dict, Any

import openai
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field

from sekai_optimizer.data.story_repository import StoryRepository
from .tools import (
    feedback_synthesis_tool,
    create_batch_evaluation_tools,
)
from .tools.ground_truth_generator import create_ground_truth_generator_tool

logger = logging.getLogger(__name__)


# Pydantic model for structured output
class EvaluationResult(BaseModel):
    """Structured output for the evaluation agent."""

    score: float = Field(
        description="The mean P@10 score across all users in the batch"
    )
    avg_semantic_score: float = Field(
        description="The mean semantic similarity score across all users"
    )
    feedback: str = Field(
        description="Synthesized, actionable feedback for prompt optimization"
    )


class FinalEvaluationResult(TypedDict):
    """The final output contract for the evaluation agent."""

    score: float  # Mean P@10 score
    feedback: str  # Synthesized, actionable feedback
    batch_size: int
    avg_semantic_score: float


def create_evaluation_agent_tools(
    story_repository: StoryRepository,
    openai_client: openai.OpenAI,
) -> List:
    """Creates the tool kit for the Evaluation Agent for batch processing."""

    # Create batch processing tools (no longer includes ground truth generator)
    batch_tools = create_batch_evaluation_tools(story_repository, openai_client)

    # Create the unified ground truth generator tool (supports both single and batch)
    ground_truth_tool = create_ground_truth_generator_tool(story_repository)

    return batch_tools + [ground_truth_tool, feedback_synthesis_tool]


def run_evaluation_agent(
    user_batch_with_recommendations: List[Dict[str, Any]],
    story_repository: StoryRepository,
    openai_client: openai.OpenAI,
) -> FinalEvaluationResult:
    """
    Main entrypoint for the autonomous Evaluation Agent.

    The agent receives pre-generated recommendations for a batch of users and evaluates them.

    Args:
        user_batch_with_recommendations: List of dicts containing:
            - "profile": user profile string
            - "recommended_ids": list of 10 recommended story IDs
            - "user_index": index in batch (optional)
    """
    batch_size = len(user_batch_with_recommendations)
    logger.info(
        f"Starting autonomous evaluation for {batch_size} users with pre-generated recommendations."
    )

    try:
        # Create the agent's tools
        tools = create_evaluation_agent_tools(
            story_repository=story_repository,
            openai_client=openai_client,
        )

        # Create a local prompt template with agent_scratchpad
        langsmith_client = LangSmithClient(api_key=os.environ.get("LANGSMITH_API_KEY"))
        runnable = langsmith_client.pull_prompt(
            "evaluation-agent:latest", include_model=True
        )
        prompt = runnable.first  # BasePromptTemplate
        model = runnable.steps[-1]  # the bound ChatModel inside the sequence

        # Create the tool-calling agent (we'll handle structured output in post-processing)
        agent = create_tool_calling_agent(
            llm=model,
            tools=tools,
            prompt=prompt,
        )

        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=50,  # Allow enough iterations for complex evaluation
        )

        # Prepare the input for the agent
        batch_data_json = json.dumps(user_batch_with_recommendations)

        # Execute the agent
        result = agent_executor.invoke(
            {
                "agent_scratchpad": "",
                "batch_data": batch_data_json,
                "batch_size": batch_size,
            }
        )

        # Extract the structured output
        if isinstance(result.get("output"), EvaluationResult):
            eval_result = result["output"]
            return FinalEvaluationResult(
                score=eval_result.score,
                feedback=eval_result.feedback,
                batch_size=batch_size,
                avg_semantic_score=eval_result.avg_semantic_score,
            )
        else:
            # Fallback parsing if agent doesn't return Pydantic object
            try:
                agent_output = result.get("output", "{}")
                if isinstance(agent_output, str):
                    parsed_result = json.loads(agent_output)
                else:
                    parsed_result = agent_output

                return FinalEvaluationResult(
                    score=parsed_result.get("score", 0.0),
                    feedback=parsed_result.get("feedback", "No feedback generated."),
                    batch_size=batch_size,
                    avg_semantic_score=parsed_result.get("avg_semantic_score", 0.0),
                )
            except (json.JSONDecodeError, TypeError):
                # Final fallback
                return FinalEvaluationResult(
                    score=0.0,
                    feedback=f"Agent completed but returned unexpected format: {result.get('output')}",
                    batch_size=batch_size,
                    avg_semantic_score=0.0,
                )

    except Exception as e:
        logger.error(f"Critical error in evaluation agent: {e}", exc_info=True)
        return FinalEvaluationResult(
            score=0.0,
            feedback=f"Critical error during evaluation: {str(e)}",
            batch_size=batch_size,
            avg_semantic_score=0.0,
        )


if __name__ == "__main__":
    # Configure logging for readability
    log_format = "[%(asctime)s] [%(levelname)-8s] [%(name)-25s] --- %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    # Silence overly verbose library loggers to focus on application logs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)

    from dotenv import load_dotenv
    from pathlib import Path
    from sekai_optimizer.data.story_repository import InMemoryStoryRepository

    load_dotenv()

    # Note: LANGCHAIN_TRACING_V2, LANGSMITH_API_KEY, and OPENAI_API_KEY must be in .env
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    logger.info("Starting Evaluation Agent Test...")

    # Initialize OpenAI client
    openai_client = openai.OpenAI()

    # Set up data paths
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DATA_DIR = PROJECT_ROOT / "src" / "sekai_optimizer" / "data"

    # Initialize story repository
    stories_repo = InMemoryStoryRepository(stories_filepath=DATA_DIR / "stories.json")
    logger.debug(
        f"Loaded story repository with {len(stories_repo.get_all_stories())} stories"
    )
    test_user_batch = [
        {
            "profile": "Adventure-seeker who loves fantasy worlds and epic quests. Enjoys stories with dragons, magic, and heroic aspirations. Prefers narratives with strong moral themes and character growth. Favors settings in mythical lands and ancient kingdoms. Enthusiastic about epic battles and tales of redemption. Fan of Tolkien, C.S. Lewis, and Brandon Sanderson.",
            "user_index": 0,
            "recommendation_ids": [
                100030,  # The Forbidden Grimoire
                100028,  # The Enigma of the Lost City
                100029,  # The Goblin King's Bargain
                100060,  # The Guardian of the Forest
                100012,  # The Arcane Heist
                100004,  # Digital Heroine: Code Break
                100062,  # The Time Traveler's Paradox
                100009,  # The Timekeeper's Curse
                100061,  # The Shadow Puppeteer's Game
                100059,  # The Midnight Thief
            ],
        },
        {
            "profile": "Romantic at heart with a penchant for historical romance and forbidden love. Prefers narratives with strong female leads and complex relationships. Enjoys settings in the Victorian era, medieval castles, and fantasy realms. Loves slow-burn romances, love triangles, and dramatic revelations. Fan of Jane Austen, Philippa Gregory, and Diana Gabaldon.",
            "user_index": 1,
            "recommendation_ids": [
                100001,  # The Vampire's Embrace
                100002,  # Forbidden Love in Medieval Times
                100003,  # The Duke's Secret Romance
                100005,  # The Princess's Dilemma
                100006,  # Love Beyond Time
                100007,  # The Enchanted Castle
                100008,  # Hearts in the Highlands
                100010,  # The Royal Affair
                100011,  # Secrets of the Manor
                100013,  # The Lady's Choice
            ],
        },
        {
            "profile": "Tech-savvy gamer who loves stories set in virtual realities and digital worlds. Enjoys narratives with hacker protagonists and AI companions. Prefers sci-fi and cyberpunk genres with elements of mystery and adventure. Favors stories with plot twists, futuristic technology, and moral dilemmas. Fan of William Gibson, Neal Stephenson, and Ernest Cline.",
            "user_index": 2,
            "recommendation_ids": [
                100004,  # Digital Heroine: Code Break
                100062,  # The Time Traveler's Paradox
                100009,  # The Timekeeper's Curse
                100061,  # The Shadow Puppeteer's Game
                100012,  # The Arcane Heist
                100059,  # The Midnight Thief
                100030,  # The Forbidden Grimoire
                100028,  # The Enigma of the Lost City
                100029,  # The Goblin King's Bargain
                100060,  # The Guardian of the Forest
            ],
        },
    ]

    try:
        # Run the evaluation agent with the batch data
        logger.info("\n--- Executing Evaluation Agent ---")
        logger.debug(
            "Agent will evaluate pre-generated recommendations for all users..."
        )

        result = run_evaluation_agent(
            user_batch_with_recommendations=test_user_batch,
            story_repository=stories_repo,
            openai_client=openai_client,
        )

        logger.info("\n--- Evaluation Results ---")
        logger.info(f"Batch Size: {result['batch_size']} users")
        logger.info(f"Average P@10 Score: {result['score']:.3f}")
        logger.info(f"Average Semantic Score: {result['avg_semantic_score']:.3f}")
        logger.info("Synthesized Feedback:")
        logger.info(f"  {result['feedback']}")

        logger.info("\n--- Test Summary ---")
        logger.debug(
            "✅ Evaluation agent successfully processed user batch with pre-generated recommendations"
        )
        logger.debug(
            "✅ This simulates the expected LangGraph flow: Recommend → Evaluate → Optimize"
        )

    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}", exc_info=True)
        logger.error("Evaluation test failed.")

    logger.info("\n--- Evaluation Agent Test Complete ---")
