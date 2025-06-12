import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the refactored functions
from sekai_optimizer.scripts import synthesize_data
from sekai_optimizer.data.types import (
    Dataset,
    Story,
    User,
    UsersResponse,
    StoriesResponse,
)

# Mock data for testing
MOCK_STORIES_DICTS = [
    {"id": 1, "title": "Mock Story 1", "intro": "Intro 1", "tags": ["tag1", "tag2"]},
    {"id": 2, "title": "Mock Story 2", "intro": "Intro 2", "tags": ["tag3", "tag4"]},
]

MOCK_USERS_DICTS = [
    {"user_id": 101, "name": "Mock User 1", "profile": "Profile 1"},
    {"user_id": 102, "name": "Mock User 2", "profile": "Profile 2"},
]

# Mock Pydantic models
MOCK_STORIES = [Story(**story) for story in MOCK_STORIES_DICTS]  # type: ignore
MOCK_USERS = [User(**user) for user in MOCK_USERS_DICTS]  # type: ignore
MOCK_DATASET = Dataset(stories=MOCK_STORIES, users=MOCK_USERS)


@pytest.fixture
def mock_langsmith_client():
    """Fixture to mock the LangSmith client and its prompt pulling."""
    with patch(
        "sekai_optimizer.scripts.synthesize_data.LangSmithClient"
    ) as mock_client_class:
        mock_instance = MagicMock()

        # Mock the pull_prompt method to return a mock runnable
        mock_runnable = MagicMock()
        mock_instance.pull_prompt.return_value = mock_runnable

        mock_client_class.return_value = mock_instance
        yield mock_instance


def test_generate_users():
    """
    Tests the generate_users function with LangSmith prompt pulling.
    """
    with patch(
        "sekai_optimizer.scripts.synthesize_data.JsonOutputParser"
    ) as mock_parser_class:
        # Create a mock chain that returns our test data when invoked
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"users": MOCK_USERS_DICTS}

        # Setup the mock parser
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser

        # Create a mock runnable and set up the chain behavior (runnable | parser)
        mock_runnable = MagicMock()
        mock_runnable.__or__.return_value = mock_chain

        # Create a mock LangSmith client
        mock_langsmith_client = MagicMock()
        mock_langsmith_client.pull_prompt.return_value = mock_runnable

        # Call the function
        result = synthesize_data.generate_users(mock_langsmith_client)

        # Assertions
        mock_langsmith_client.pull_prompt.assert_called_once_with(
            "data-synthesis-users:latest", include_model=True
        )
        mock_parser_class.assert_called_once_with(pydantic_object=UsersResponse)
        mock_runnable.__or__.assert_called_once_with(mock_parser)
        mock_chain.invoke.assert_called_once_with({})

        assert result == MOCK_USERS_DICTS
        assert len(result) == 2


def test_generate_stories():
    """
    Tests the generate_stories function with LangSmith prompt pulling.
    """
    with patch(
        "sekai_optimizer.scripts.synthesize_data.JsonOutputParser"
    ) as mock_parser_class:
        # Create a mock chain that returns our test data when invoked
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"stories": MOCK_STORIES_DICTS}

        # Setup the mock parser
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser

        # Create a mock runnable and set up the chain behavior (runnable | parser)
        mock_runnable = MagicMock()
        mock_runnable.__or__.return_value = mock_chain

        # Create a mock LangSmith client
        mock_langsmith_client = MagicMock()
        mock_langsmith_client.pull_prompt.return_value = mock_runnable

        # Call the function
        result = synthesize_data.generate_stories(mock_langsmith_client)

        # Assertions
        mock_langsmith_client.pull_prompt.assert_called_once_with(
            "data-synthesis-stories:latest", include_model=True
        )
        mock_parser_class.assert_called_once_with(pydantic_object=StoriesResponse)
        mock_runnable.__or__.assert_called_once_with(mock_parser)
        mock_chain.invoke.assert_called_once_with({})

        assert result == MOCK_STORIES_DICTS
        assert len(result) == 2


def test_synthesize_and_save_data_integration(mock_langsmith_client, tmp_path):
    """
    Integration test for the main synthesize_and_save_data function.
    Mocks both generate_users and generate_stories functions.
    """
    stories_path = tmp_path / "stories.json"
    users_path = tmp_path / "users.json"

    # Mock the individual generation functions
    with (
        patch(
            "sekai_optimizer.scripts.synthesize_data.generate_users"
        ) as mock_gen_users,
        patch(
            "sekai_optimizer.scripts.synthesize_data.generate_stories"
        ) as mock_gen_stories,
    ):

        mock_gen_users.return_value = MOCK_USERS_DICTS
        mock_gen_stories.return_value = MOCK_STORIES_DICTS

        # Call the function to be tested
        synthesize_data.synthesize_and_save_data(
            langsmith_client=mock_langsmith_client,
            stories_filepath=stories_path,
            users_filepath=users_path,
        )

        # Assert that the generation functions were called
        mock_gen_users.assert_called_once_with(mock_langsmith_client)
        mock_gen_stories.assert_called_once_with(mock_langsmith_client)

    # Assert files were created and contain correct data
    assert stories_path.exists()
    assert users_path.exists()

    with open(stories_path, "r") as f:
        saved_stories = json.load(f)
    with open(users_path, "r") as f:
        saved_users = json.load(f)

    assert saved_stories == MOCK_STORIES_DICTS
    assert saved_users == MOCK_USERS_DICTS

    print(
        "\nTest passed: `synthesize_and_save_data` correctly used LangSmith prompts and saved the data."
    )


def test_synthesize_and_save_data_creates_directories(mock_langsmith_client, tmp_path):
    """
    Tests that synthesize_and_save_data creates parent directories if they don't exist.
    """
    # Create nested path that doesn't exist
    nested_dir = tmp_path / "deep" / "nested" / "path"
    stories_path = nested_dir / "stories.json"
    users_path = nested_dir / "users.json"

    # Mock the individual generation functions
    with (
        patch(
            "sekai_optimizer.scripts.synthesize_data.generate_users"
        ) as mock_gen_users,
        patch(
            "sekai_optimizer.scripts.synthesize_data.generate_stories"
        ) as mock_gen_stories,
    ):

        mock_gen_users.return_value = MOCK_USERS_DICTS
        mock_gen_stories.return_value = MOCK_STORIES_DICTS

        # Call the function
        synthesize_data.synthesize_and_save_data(
            langsmith_client=mock_langsmith_client,
            stories_filepath=stories_path,
            users_filepath=users_path,
        )

    # Assert directories were created and files exist
    assert nested_dir.exists()
    assert stories_path.exists()
    assert users_path.exists()
