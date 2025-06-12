"""
Unit tests for pick_users_node implementation.

Following TDD approach for Stage 2 Step 2.1.
"""

import time

from sekai_optimizer.nodes.pick_users import pick_users_node
from sekai_optimizer.data.state import OptimizationState


class TestPickUsersNode:
    """Test suite for pick_users_node functionality."""

    def test_normal_case_random_selection(self):
        """Test normal case: select random batch smaller than total users."""
        # Arrange: Create mock state with 5 users, request 3
        mock_users = [
            {"user_id": 1, "name": "Alice", "profile": "Adventure lover"},
            {"user_id": 2, "name": "Bob", "profile": "Romance reader"},
            {"user_id": 3, "name": "Charlie", "profile": "Sci-fi fan"},
            {"user_id": 4, "name": "Diana", "profile": "Mystery enthusiast"},
            {"user_id": 5, "name": "Eve", "profile": "Fantasy lover"},
        ]

        state: OptimizationState = {
            "full_training_users": mock_users,
            "config": {"user_per_batch": 3},
            # Required state fields (not used by this node)
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        # Act: Call the function
        result = pick_users_node(state)

        # Assert: Check the result format and size
        assert "current_user_batch" in result
        assert isinstance(result["current_user_batch"], list)
        assert len(result["current_user_batch"]) == 3

        # Assert: Each selected user should be from the original list
        selected_users = result["current_user_batch"]
        for user in selected_users:
            assert user in mock_users

        # Assert: Should be unique users (no duplicates)
        selected_ids = [user["user_id"] for user in selected_users]
        assert len(selected_ids) == len(set(selected_ids))

    def test_batch_size_equals_total_users(self):
        """Test edge case: batch size equals total number of users."""
        mock_users = [
            {"user_id": 1, "name": "Alice", "profile": "Adventure lover"},
            {"user_id": 2, "name": "Bob", "profile": "Romance reader"},
            {"user_id": 3, "name": "Charlie", "profile": "Sci-fi fan"},
        ]

        state: OptimizationState = {
            "full_training_users": mock_users,
            "config": {"user_per_batch": 3},
            # Required state fields
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        result = pick_users_node(state)

        # Should return all users
        assert len(result["current_user_batch"]) == 3
        selected_ids = [user["user_id"] for user in result["current_user_batch"]]
        original_ids = [user["user_id"] for user in mock_users]
        assert set(selected_ids) == set(original_ids)

    def test_batch_size_larger_than_total_users(self):
        """Test edge case: batch size larger than total number of users."""
        mock_users = [
            {"user_id": 1, "name": "Alice", "profile": "Adventure lover"},
            {"user_id": 2, "name": "Bob", "profile": "Romance reader"},
        ]

        state: OptimizationState = {
            "full_training_users": mock_users,
            "config": {"user_per_batch": 5},  # Requesting more than available
            # Required state fields
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        result = pick_users_node(state)

        # Should return all available users (not more)
        assert len(result["current_user_batch"]) == 2
        selected_ids = [user["user_id"] for user in result["current_user_batch"]]
        original_ids = [user["user_id"] for user in mock_users]
        assert set(selected_ids) == set(original_ids)

    def test_empty_user_list(self):
        """Test edge case: empty user list."""
        state: OptimizationState = {
            "full_training_users": [],
            "config": {"user_per_batch": 3},
            # Required state fields
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        result = pick_users_node(state)

        # Should return empty list
        assert result["current_user_batch"] == []

    def test_batch_size_zero(self):
        """Test edge case: batch size is zero."""
        mock_users = [
            {"user_id": 1, "name": "Alice", "profile": "Adventure lover"},
            {"user_id": 2, "name": "Bob", "profile": "Romance reader"},
        ]

        state: OptimizationState = {
            "full_training_users": mock_users,
            "config": {"user_per_batch": 0},
            # Required state fields
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        result = pick_users_node(state)

        # Should return empty list
        assert result["current_user_batch"] == []

    def test_single_user_selection(self):
        """Test normal case: select single user from multiple."""
        mock_users = [
            {"user_id": 1, "name": "Alice", "profile": "Adventure lover"},
            {"user_id": 2, "name": "Bob", "profile": "Romance reader"},
            {"user_id": 3, "name": "Charlie", "profile": "Sci-fi fan"},
        ]

        state: OptimizationState = {
            "full_training_users": mock_users,
            "config": {"user_per_batch": 1},
            # Required state fields
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        result = pick_users_node(state)

        # Should return exactly one user
        assert len(result["current_user_batch"]) == 1
        selected_user = result["current_user_batch"][0]
        assert selected_user in mock_users

    def test_randomness_with_multiple_calls(self):
        """Test that multiple calls with same state produce different results (randomness)."""
        mock_users = [
            {"user_id": i, "name": f"User{i}", "profile": f"Profile {i}"}
            for i in range(1, 11)  # 10 users
        ]

        state: OptimizationState = {
            "full_training_users": mock_users,
            "config": {"user_per_batch": 3},
            # Required state fields
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        # Make multiple calls and collect results
        results = []
        for _ in range(10):
            result = pick_users_node(state)
            selected_ids = tuple(
                sorted([user["user_id"] for user in result["current_user_batch"]])
            )
            results.append(selected_ids)

        # Should get some variation (not all identical)
        # Note: There's a small chance all results are identical, but very unlikely with 10 users choose 3
        unique_results = set(results)
        assert (
            len(unique_results) > 1
        ), "Expected some randomness in selection, but got identical results"

    def test_actual_data_format_compliance(self):
        """Test with actual data format from users.json to ensure compliance."""
        # Use actual data format from the data files
        actual_format_users = [
            {
                "user_id": 1,
                "name": "Aiden",
                "profile": "Adventure-seeker who loves fantasy worlds and epic quests. Enjoys stories with dragons, magic, and heroic aspirations.",
            },
            {
                "user_id": 2,
                "name": "Sophia",
                "profile": "Romantic at heart with a penchant for historical romance and forbidden love. Prefers narratives with strong female leads.",
            },
            {
                "user_id": 3,
                "name": "Liam",
                "profile": "Tech-savvy gamer who loves stories set in virtual realities and digital worlds. Enjoys narratives with hacker protagonists.",
            },
        ]

        state: OptimizationState = {
            "full_training_users": actual_format_users,
            "config": {"user_per_batch": 2},
            # Required state fields
            "current_user_batch": [],
            "current_strategy_prompt": "",
            "iteration_count": 0,
            "batch_simulated_tags": {},
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "start_time": time.time(),
            "all_stories": [],
            "faiss_index_path": "",
        }

        result = pick_users_node(state)

        # Verify result format matches expectations
        assert len(result["current_user_batch"]) == 2
        for user in result["current_user_batch"]:
            assert "user_id" in user
            assert "name" in user
            assert "profile" in user
            assert isinstance(user["user_id"], int)
            assert isinstance(user["name"], str)
            assert isinstance(user["profile"], str)
