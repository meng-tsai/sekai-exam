import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Assume the function will be in this module
from seikai_optimizer.scripts import synthesize_data
from seikai_optimizer.data.types import Dataset, Story, User

# Mock Pydantic models
MOCK_STORIES = [
    Story(id=1, title="Mock Story 1", intro="Intro 1", tags=["tag1", "tag2"]),
    Story(id=2, title="Mock Story 2", intro="Intro 2", tags=["tag3", "tag4"]),
]
MOCK_USERS = [
    User(user_id=101, name="Mock User 1", profile="Profile 1"),
    User(user_id=102, name="Mock User 2", profile="Profile 2"),
]
MOCK_DATASET = Dataset(stories=MOCK_STORIES, users=MOCK_USERS)


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client and its .parse() response."""
    with patch("openai.OpenAI") as mock_client_class:
        mock_instance = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        # This is the key change: mock the .parsed attribute
        mock_message.parsed = MOCK_DATASET
        mock_message.refusal = None

        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]

        # Mock the new method call
        mock_instance.beta.chat.completions.parse.return_value = mock_completion

        mock_client_class.return_value = mock_instance
        yield mock_instance


def test_synthesize_and_save_data_with_pydantic(mock_openai_client, tmp_path):
    """
    Tests the main data synthesis function with Pydantic model parsing.
    - Mocks the OpenAI API call to `beta.chat.completions.parse`.
    - Verifies that the function saves the data from the Pydantic models correctly.
    """
    stories_path = tmp_path / "stories.json"
    users_path = tmp_path / "users.json"

    # Call the function to be tested
    synthesize_data.synthesize_and_save_data(
        client=mock_openai_client,
        stories_filepath=stories_path,
        users_filepath=users_path,
    )

    # 1. Assert LLM was called
    mock_openai_client.beta.chat.completions.parse.assert_called_once()

    # 2. Assert files were created and contain correct data
    assert stories_path.exists()
    assert users_path.exists()

    with open(stories_path, "r") as f:
        saved_stories = json.load(f)
    with open(users_path, "r") as f:
        saved_users = json.load(f)

    # Convert mock Pydantic models to dicts for comparison
    expected_stories = [s.model_dump() for s in MOCK_STORIES]
    expected_users = [u.model_dump() for u in MOCK_USERS]

    assert saved_stories == expected_stories
    assert saved_users == expected_users

    print(
        "\nTest passed: `synthesize_and_save_data` correctly used Pydantic parsing and saved the data."
    )
