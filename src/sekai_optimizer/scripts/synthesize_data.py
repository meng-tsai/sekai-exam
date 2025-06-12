import os
import json
import logging
from pathlib import Path

import openai
from dotenv import load_dotenv
from langsmith import Client as LangSmithClient
from langchain_core.output_parsers import JsonOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from sekai_optimizer.data.types import (
    Dataset,
    Story,
    User,
    UsersResponse,
    StoriesResponse,
)


def generate_users(langsmith_client: LangSmithClient) -> list[dict]:
    """
    Generates synthetic user data using LangSmith prompt with structured output.

    Args:
        langsmith_client: An initialized LangSmith client.

    Returns:
        A list of user dictionaries.
    """
    logger.debug("Generating synthetic user data using LangSmith prompt...")

    # Pull the user generation prompt from LangSmith
    user_prompt_runnable = langsmith_client.pull_prompt(
        "data-synthesis-users:latest", include_model=True
    )

    json_parser = JsonOutputParser(pydantic_object=UsersResponse)
    chain = user_prompt_runnable | json_parser

    users = chain.invoke({}).get("users", [])
    logger.info(f"Successfully generated {len(users)} user profiles")
    return users


def generate_stories(langsmith_client: LangSmithClient) -> list[dict]:
    """
    Generates synthetic story data using LangSmith prompt with structured output.

    Args:
        langsmith_client: An initialized LangSmith client.

    Returns:
        A list of story dictionaries.
    """
    logger.debug("Generating synthetic story data using LangSmith prompt...")

    # Pull the story generation prompt from LangSmith
    story_prompt_runnable = langsmith_client.pull_prompt(
        "data-synthesis-stories:latest", include_model=True
    )

    json_parser = JsonOutputParser(pydantic_object=StoriesResponse)
    chain = story_prompt_runnable | json_parser

    stories = chain.invoke({}).get("stories", [])

    logger.info(f"Successfully generated {len(stories)} story concepts")
    return stories


def synthesize_and_save_data(
    langsmith_client: LangSmithClient,
    stories_filepath: Path,
    users_filepath: Path,
) -> None:
    """
    Generates synthetic story and user data using LangSmith prompts and saves it to files.

    Args:
        langsmith_client: An initialized LangSmith client.
        stories_filepath: The path to save the stories JSON file.
        users_filepath: The path to save the users JSON file.
    """
    logger.debug("Generating synthetic data using LangSmith prompts...")

    # Generate users and stories separately
    users = generate_users(langsmith_client)
    stories = generate_stories(langsmith_client)

    # Ensure parent directories exist
    stories_filepath.parent.mkdir(parents=True, exist_ok=True)
    users_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save the data to the specified files
    with open(stories_filepath, "w") as f:
        json.dump(stories, f, indent=2)
    logger.info(f"Successfully saved {len(stories)} stories to {stories_filepath}")

    with open(users_filepath, "w") as f:
        json.dump(users, f, indent=2)
    logger.info(f"Successfully saved {len(users)} users to {users_filepath}")


if __name__ == "__main__":
    # This allows the script to be run directly to generate the data
    load_dotenv()

    # Check for required API keys
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError(
            "LANGSMITH_API_KEY not found in .env file or environment variables."
        )

    # Define file paths
    # The script is in src/sekai_optimizer/scripts, so we go up three levels for the root
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "src" / "sekai_optimizer" / "data"

    stories_file = DATA_DIR / "stories.json"
    users_file = DATA_DIR / "users.json"

    # Initialize LangSmith client and run synthesis
    langsmith_client = LangSmithClient(api_key=os.environ.get("LANGSMITH_API_KEY"))
    synthesize_and_save_data(langsmith_client, stories_file, users_file)
    logger.info("Data synthesis complete.")
