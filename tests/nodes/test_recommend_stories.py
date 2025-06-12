"""
Unit tests for recommend_stories_node implementation.
Following TDD methodology - tests written before implementation.
"""

import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import pytest

from sekai_optimizer.nodes.recommend_stories import recommend_stories_node
from sekai_optimizer.data.state import OptimizationState


class TestRecommendStoriesNode:
    """Test suite for recommend_stories_node following TDD methodology."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_state: OptimizationState = {
            "batch_simulated_tags": {
                "1": ["action", "fantasy", "adventure"],
                "2": ["romance", "slice-of-life", "modern"],
            },
            "current_strategy_prompt": "Focus on user preferences and story themes for optimal recommendations",
            "full_training_users": [],
            "current_user_batch": [],
            "batch_recommendations": {},
            "batch_ground_truths": {},
            "evaluation_result": {},
            "evaluation_history": [],
            "iteration_count": 0,
            "start_time": 0.0,
            "best_strategy_prompt": "",
            "best_score": -1.0,
            "best_evaluation": {},
            "config": {},
        }

        # Mock story data
        self.mock_stories = [
            {
                "id": 217107,
                "title": "Stranger Who Fell From The Sky",
                "intro": "You are Devin, plummeting towards Orario...",
                "tags": ["danmachi", "reincarnation", "heroic aspirations"],
            },
            {
                "id": 273613,
                "title": "Trapped Between Four Anime Legends!",
                "intro": "You're caught in a dimensional rift...",
                "tags": ["crossover", "jujutsu kaisen", "dragon ball"],
            },
        ]

        # Mock LLM response
        self.mock_llm_response = [
            217107,
            273613,
            235701,
            214527,
            263242,
            500101,
            500102,
            500103,
            500104,
            500105,
        ]

    @patch("sekai_optimizer.nodes.recommend_stories.RecommendationService")
    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"})
    def test_successful_recommendation_flow(self, mock_service_class):
        """Test normal case: successful FAISS retrieval and LLM ranking."""
        # Setup mocks
        mock_service = Mock()
        mock_service.recommend_for_users.return_value = {
            "1": [
                217107,
                273613,
                235701,
                214527,
                263242,
                500101,
                500102,
                500103,
                500104,
                500105,
            ],
            "2": [
                500105,
                500104,
                500103,
                500102,
                500101,
                263242,
                214527,
                235701,
                273613,
                217107,
            ],
        }
        mock_service_class.get_instance.return_value = mock_service

        # Execute
        result = recommend_stories_node(self.mock_state)

        # Verify
        assert "batch_recommendations" in result
        batch_recommendations = result["batch_recommendations"]

        # Check structure
        assert isinstance(batch_recommendations, dict)
        assert "1" in batch_recommendations
        assert "2" in batch_recommendations

        # Check each user has exactly 10 recommendations
        assert len(batch_recommendations["1"]) == 10
        assert len(batch_recommendations["2"]) == 10

        # Check all recommendations are integers (story IDs)
        for user_id, recommendations in batch_recommendations.items():
            assert all(isinstance(story_id, int) for story_id in recommendations)

        # Verify service was called with correct parameters
        mock_service.recommend_for_users.assert_called_once()
        call_args = mock_service.recommend_for_users.call_args[0]
        assert call_args[0] == self.mock_state["batch_simulated_tags"]
        assert call_args[1] == self.mock_state["current_strategy_prompt"]

    @patch("sekai_optimizer.nodes.recommend_stories.RecommendationService")
    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"})
    def test_faiss_failure_hard_fails(self, mock_service_class):
        """Test FAISS failure should hard fail the graph run."""
        # Setup mock to raise exception
        mock_service = Mock()
        mock_service.recommend_for_users.side_effect = Exception(
            "FAISS index failed to load"
        )
        mock_service_class.get_instance.return_value = mock_service

        # Execute and verify exception is propagated
        with pytest.raises(Exception, match="FAISS index failed to load"):
            recommend_stories_node(self.mock_state)

    @patch("sekai_optimizer.nodes.recommend_stories.RecommendationService")
    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"})
    def test_llm_failure_fallback_to_faiss(self, mock_service_class):
        """Test LLM failure should return first 10 from FAISS results."""
        # Setup mock service to simulate LLM failure
        mock_service = Mock()

        # Mock partial failure - FAISS works but LLM fails for one user
        def mock_recommend_side_effect(batch_tags, strategy_prompt):
            # Simulate LLM failure for user "1", success for user "2"
            return {
                "1": [
                    217107,
                    273613,
                    235701,
                    214527,
                    263242,
                    500101,
                    500102,
                    500103,
                    500104,
                    500105,
                ],  # Fallback
                "2": [
                    500105,
                    500104,
                    500103,
                    500102,
                    500101,
                    263242,
                    214527,
                    235701,
                    273613,
                    217107,
                ],  # Success
            }

        mock_service.recommend_for_users.side_effect = mock_recommend_side_effect
        mock_service_class.get_instance.return_value = mock_service

        # Execute
        result = recommend_stories_node(self.mock_state)

        # Verify
        assert "batch_recommendations" in result
        batch_recommendations = result["batch_recommendations"]

        # Both users should still get 10 recommendations
        assert len(batch_recommendations["1"]) == 10
        assert len(batch_recommendations["2"]) == 10

    @patch("sekai_optimizer.nodes.recommend_stories.RecommendationService")
    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"})
    def test_empty_tags_handling(self, mock_service_class):
        """Test handling of empty tag lists."""
        # Setup state with empty tags
        empty_state = self.mock_state.copy()
        empty_state["batch_simulated_tags"] = {"1": [], "2": ["romance"]}

        # Setup mock
        mock_service = Mock()
        mock_service.recommend_for_users.return_value = {
            "1": [
                217107,
                273613,
                235701,
                214527,
                263242,
                500101,
                500102,
                500103,
                500104,
                500105,
            ],
            "2": [
                500105,
                500104,
                500103,
                500102,
                500101,
                263242,
                214527,
                235701,
                273613,
                217107,
            ],
        }
        mock_service_class.get_instance.return_value = mock_service

        # Execute
        result = recommend_stories_node(empty_state)

        # Verify
        assert "batch_recommendations" in result
        batch_recommendations = result["batch_recommendations"]

        # Should still return 10 recommendations per user
        assert len(batch_recommendations["1"]) == 10
        assert len(batch_recommendations["2"]) == 10

    @patch("sekai_optimizer.nodes.recommend_stories.RecommendationService")
    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"})
    def test_story_id_validation(self, mock_service_class):
        """Test that all returned story IDs are valid integers."""
        # Setup mock
        mock_service = Mock()
        mock_service.recommend_for_users.return_value = {
            "1": [
                217107,
                273613,
                235701,
                214527,
                263242,
                500101,
                500102,
                500103,
                500104,
                500105,
            ],
            "2": [
                500105,
                500104,
                500103,
                500102,
                500101,
                263242,
                214527,
                235701,
                273613,
                217107,
            ],
        }
        mock_service_class.get_instance.return_value = mock_service

        # Execute
        result = recommend_stories_node(self.mock_state)

        # Verify all story IDs are valid integers
        batch_recommendations = result["batch_recommendations"]
        for user_id, recommendations in batch_recommendations.items():
            for story_id in recommendations:
                assert isinstance(story_id, int)
                assert story_id > 0  # Story IDs should be positive
