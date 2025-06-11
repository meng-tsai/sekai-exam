"""
End-to-end test for the Sekai Optimizer workflow.

This test validates the complete workflow execution with stub implementations,
ensuring that the state is passed and updated correctly through all nodes.
"""

import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.sekai_optimizer.workflows.graph_builder import build_optimization_graph
from src.sekai_optimizer.workflows.state_initialization import (
    initialize_optimization_state,
)


@pytest.fixture
def mock_data_files():
    """Create temporary data files for testing."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create mock stories data
    mock_stories = {
        "stories": [
            {
                "id": 1,
                "title": "Test Story 1",
                "intro": "A test story about adventure",
                "tags": ["action", "adventure"],
            },
            {
                "id": 2,
                "title": "Test Story 2",
                "intro": "A test story about romance",
                "tags": ["romance", "drama"],
            },
            {
                "id": 3,
                "title": "Test Story 3",
                "intro": "A test story about comedy",
                "tags": ["comedy", "slice-of-life"],
            },
        ]
        + [
            {
                "id": i,
                "title": f"Test Story {i}",
                "intro": f"Test story number {i}",
                "tags": ["test", "mock"],
            }
            for i in range(4, 21)  # Create 20 total stories for testing
        ]
    }

    # Create mock users data
    mock_users = {
        "users": [
            {
                "user_id": "user_1",
                "name": "Test User 1",
                "profile": "A test user who likes action and adventure stories",
            },
            {
                "user_id": "user_2",
                "name": "Test User 2",
                "profile": "A test user who enjoys romance and drama",
            },
            {
                "user_id": "user_3",
                "name": "Test User 3",
                "profile": "A test user who prefers comedy and slice-of-life",
            },
            {
                "user_id": "user_4",
                "name": "Test User 4",
                "profile": "A test user with varied interests",
            },
            {
                "user_id": "user_5",
                "name": "Test User 5",
                "profile": "Another test user for batch testing",
            },
        ]
    }

    # Write files
    stories_file = temp_path / "stories.json"
    users_file = temp_path / "users.json"

    with open(stories_file, "w") as f:
        json.dump(mock_stories, f)

    with open(users_file, "w") as f:
        json.dump(mock_users, f)

    return temp_path, stories_file, users_file


class TestE2EWorkflow:
    """End-to-end test suite for the optimization workflow."""

    @patch.dict(
        os.environ,
        {
            "USER_PER_BATCH": "3",
            "MAX_ITERATIONS": "2",
            "MAX_OPTIMIZATION_MINUTE": "10",
            "INIT_STRATEGY_PROMPT": "Test strategy prompt",
        },
    )
    def test_complete_workflow_execution(self, mock_data_files):
        """Test that the complete workflow executes successfully with stub implementations."""
        temp_path, stories_file, users_file = mock_data_files

        # Load the actual mock data from the files
        with open(users_file, "r") as f:
            test_users = json.load(f)["users"]
        with open(stories_file, "r") as f:
            test_stories = json.load(f)["stories"]

        # Mock the data loading function to return our test data
        with patch(
            "src.sekai_optimizer.workflows.state_initialization.load_training_data"
        ) as mock_load_data:
            # Return our mock data
            mock_load_data.return_value = (test_users, test_stories)

            # Initialize state
            initial_state = initialize_optimization_state()

            # Validate initial state
            assert initial_state is not None
            assert len(initial_state["full_training_users"]) == 5
            assert len(initial_state["all_stories"]) == 20
            assert initial_state["iteration_count"] == 0
            assert initial_state["current_strategy_prompt"] == "Test strategy prompt"
            assert initial_state["config"]["user_per_batch"] == 3
            assert initial_state["config"]["max_iterations"] == 2

            # Build and execute graph
            graph = build_optimization_graph()
            final_state = graph.invoke(initial_state)

            # Validate final state
            self._validate_final_state(final_state)

    def _validate_final_state(self, final_state):
        """Validate the final state contains expected data."""
        # Basic state validation
        assert final_state is not None
        assert "iteration_count" in final_state
        assert "evaluation_history" in final_state
        assert "current_strategy_prompt" in final_state

        # Best result tracking validation (NEW)
        assert "best_strategy_prompt" in final_state
        assert "best_score" in final_state
        assert "best_evaluation" in final_state

        # Ensure we have some iterations
        assert final_state["iteration_count"] > 0
        assert len(final_state["evaluation_history"]) > 0

        # Best score should be >= -1 (our initial value)
        assert final_state["best_score"] >= -1.0

        # Best prompt should exist
        assert len(final_state["best_strategy_prompt"]) > 0

        # State should contain batch data from the last iteration
        assert "current_user_batch" in final_state
        assert "batch_simulated_tags" in final_state
        assert "batch_recommendations" in final_state
        assert "batch_ground_truths" in final_state

        print(
            f"✅ Validation passed - Final iteration: {final_state['iteration_count']}"
        )
        print(f"✅ Best score achieved: {final_state['best_score']}")
        print(f"✅ Best prompt: {final_state['best_strategy_prompt'][:100]}...")

    @patch.dict(
        os.environ,
        {"USER_PER_BATCH": "2", "MAX_ITERATIONS": "1", "MAX_OPTIMIZATION_MINUTE": "1"},
    )
    def test_stopping_conditions(self, mock_data_files):
        """Test that workflow respects stopping conditions."""
        temp_path, stories_file, users_file = mock_data_files

        # Load the actual mock data from the files
        with open(users_file, "r") as f:
            test_users = json.load(f)["users"]
        with open(stories_file, "r") as f:
            test_stories = json.load(f)["stories"]

        # Mock the data loading function to return our test data
        with patch(
            "src.sekai_optimizer.workflows.state_initialization.load_training_data"
        ) as mock_load_data:
            # Return our mock data
            mock_load_data.return_value = (test_users, test_stories)

            initial_state = initialize_optimization_state()
            graph = build_optimization_graph()

            final_state = graph.invoke(initial_state)

            # Should stop after 1 iteration due to MAX_ITERATIONS=1
            assert final_state["iteration_count"] == 1
            assert len(final_state["evaluation_history"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
