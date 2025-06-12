"""
Unit tests for generate_groundtruths_node following TDD methodology.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langsmith import Client as LangSmithClient

from sekai_optimizer.nodes.generate_groundtruths import (
    generate_groundtruths_node,
    GroundTruthResponse,
    _generate_ground_truth_for_user,
)
from sekai_optimizer.data.state import OptimizationState


class TestGenerateGroundtruthsNode:
    """Test suite for generate_groundtruths_node with comprehensive coverage."""

    @pytest.fixture
    def sample_state(self):
        """Sample state with user batch."""
        return {
            "current_user_batch": [
                {
                    "user_id": 1,
                    "profile": "choice-driven, high-agency, dominant protector strategist; underdog, rivalry, team-vs-team",
                },
                {
                    "user_id": 2,
                    "profile": "Self-insert choice-driven narrator as reluctant/supportive guardian, disguised royalty",
                },
            ]
        }

    @pytest.fixture
    def sample_story_data(self):
        """Sample story data for testing."""
        return {
            1: {"title": "Story 1", "intro": "Intro 1", "tags": ["tag1", "tag2"]},
            2: {"title": "Story 2", "intro": "Intro 2", "tags": ["tag3", "tag4"]},
            3: {"title": "Story 3", "intro": "Intro 3", "tags": ["tag5", "tag6"]},
            4: {"title": "Story 4", "intro": "Intro 4", "tags": ["tag7", "tag8"]},
            5: {"title": "Story 5", "intro": "Intro 5", "tags": ["tag9", "tag10"]},
            6: {"title": "Story 6", "intro": "Intro 6", "tags": ["tag11", "tag12"]},
            7: {"title": "Story 7", "intro": "Intro 7", "tags": ["tag13", "tag14"]},
            8: {"title": "Story 8", "intro": "Intro 8", "tags": ["tag15", "tag16"]},
            9: {"title": "Story 9", "intro": "Intro 9", "tags": ["tag17", "tag18"]},
            10: {"title": "Story 10", "intro": "Intro 10", "tags": ["tag19", "tag20"]},
            11: {"title": "Story 11", "intro": "Intro 11", "tags": ["tag21", "tag22"]},
            12: {"title": "Story 12", "intro": "Intro 12", "tags": ["tag23", "tag24"]},
        }

    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test_key"})
    @patch("sekai_optimizer.nodes.generate_groundtruths.RecommendationService")
    @patch("sekai_optimizer.nodes.generate_groundtruths.RunnableParallel")
    def test_normal_ground_truth_generation(
        self, mock_parallel, mock_service, sample_state, sample_story_data
    ):
        """Test normal ground truth generation flow with RunnableParallel."""
        # Setup mocks
        mock_service.get_instance.return_value.story_data = sample_story_data

        # Mock RunnableParallel to return expected results
        mock_parallel_instance = Mock()
        mock_parallel.return_value = mock_parallel_instance
        mock_parallel_instance.invoke.return_value = {
            "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }

        # Execute
        result = generate_groundtruths_node(sample_state)

        # Verify
        assert "batch_ground_truths" in result
        batch_ground_truths = result["batch_ground_truths"]

        assert len(batch_ground_truths) == 2
        assert "1" in batch_ground_truths
        assert "2" in batch_ground_truths

        # Check that each user has exactly 10 recommendations
        assert len(batch_ground_truths["1"]) == 10
        assert len(batch_ground_truths["2"]) == 10

        # Verify RunnableParallel was used
        mock_parallel.assert_called_once()
        mock_parallel_instance.invoke.assert_called_once_with({})

    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test_key"})
    @patch("sekai_optimizer.nodes.generate_groundtruths.RecommendationService")
    @patch("sekai_optimizer.nodes.generate_groundtruths.LangSmithClient")
    def test_invalid_story_ids_padding(
        self, mock_langsmith_client, mock_service, sample_state, sample_story_data
    ):
        """Test handling of invalid story IDs with padding."""
        # Setup mocks with some invalid story IDs
        mock_service.get_instance.return_value.story_data = sample_story_data

        mock_chain = Mock()
        # Return some invalid story IDs (99, 100 don't exist) and only 8 valid ones
        mock_chain.invoke.side_effect = [
            {"recommendations": [1, 2, 99, 100, 3, 4, 5, 6, 7, 8]},  # User 1
            {"recommendations": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},  # User 2
        ]

        mock_runnable = Mock()
        mock_langsmith_client.return_value.pull_prompt.return_value = mock_runnable
        mock_runnable.__or__ = Mock(return_value=mock_chain)

        # Execute
        result = generate_groundtruths_node(sample_state)

        # Verify
        batch_ground_truths = result["batch_ground_truths"]

        # Should still have exactly 10 recommendations per user
        assert len(batch_ground_truths["1"]) == 10
        assert len(batch_ground_truths["2"]) == 10

        # All recommendations should be valid story IDs
        for user_id, recommendations in batch_ground_truths.items():
            for rec in recommendations:
                assert rec in sample_story_data

    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test_key"})
    @patch("sekai_optimizer.nodes.generate_groundtruths.RecommendationService")
    @patch("sekai_optimizer.nodes.generate_groundtruths.LangSmithClient")
    def test_langsmith_connection_failure(
        self, mock_langsmith_client, mock_service, sample_state, sample_story_data
    ):
        """Test hard failure when LangSmith connection fails."""
        # Setup mocks
        mock_service.get_instance.return_value.story_data = sample_story_data
        mock_langsmith_client.return_value.pull_prompt.side_effect = Exception(
            "LangSmith connection failed"
        )

        # Execute and verify hard failure
        with pytest.raises(Exception) as exc_info:
            generate_groundtruths_node(sample_state)

        assert "LangSmith connection failed" in str(exc_info.value)

    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test_key"})
    @patch("sekai_optimizer.nodes.generate_groundtruths.RecommendationService")
    @patch("sekai_optimizer.nodes.generate_groundtruths.LangSmithClient")
    def test_llm_call_failure(
        self, mock_langsmith_client, mock_service, sample_state, sample_story_data
    ):
        """Test hard failure when LLM call fails for a user."""
        # Setup mocks
        mock_service.get_instance.return_value.story_data = sample_story_data

        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("LLM call failed")

        mock_runnable = Mock()
        mock_langsmith_client.return_value.pull_prompt.return_value = mock_runnable
        mock_runnable.__or__ = Mock(return_value=mock_chain)

        # Execute and verify hard failure
        with pytest.raises(Exception) as exc_info:
            generate_groundtruths_node(sample_state)

        assert "LLM call failed" in str(exc_info.value)

    def test_empty_user_batch(self):
        """Test handling of empty user batch."""
        empty_state = {"current_user_batch": []}

        # Should complete successfully with empty results
        with patch("sekai_optimizer.nodes.generate_groundtruths.RecommendationService"):
            result = generate_groundtruths_node(empty_state)

        assert "batch_ground_truths" in result
        assert result["batch_ground_truths"] == {}

    def test_ground_truth_response_model(self):
        """Test Pydantic model validation."""
        # Valid response
        valid_response = GroundTruthResponse(
            recommendations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        assert len(valid_response.recommendations) == 10
        assert all(isinstance(r, int) for r in valid_response.recommendations)

        # Invalid response should raise validation error
        with pytest.raises(Exception):
            GroundTruthResponse(recommendations=["invalid", "data"])

    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test_key"})
    @patch("sekai_optimizer.nodes.generate_groundtruths.RecommendationService")
    @patch("sekai_optimizer.nodes.generate_groundtruths.LangSmithClient")
    def test_story_formatting(
        self, mock_langsmith_client, mock_service, sample_state, sample_story_data
    ):
        """Test that stories are formatted correctly for the LLM prompt."""
        # Setup mocks
        mock_service.get_instance.return_value.story_data = sample_story_data

        mock_chain = Mock()
        mock_chain.invoke.return_value = {
            "recommendations": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }

        mock_runnable = Mock()
        mock_langsmith_client.return_value.pull_prompt.return_value = mock_runnable
        mock_runnable.__or__ = Mock(return_value=mock_chain)

        # Execute
        generate_groundtruths_node(sample_state)

        # Verify that invoke was called with properly formatted stories
        call_args = mock_chain.invoke.call_args_list[0][0][0]

        assert "user_profile" in call_args
        assert "complete_stories" in call_args

        # Check story formatting
        complete_stories = call_args["complete_stories"]
        assert (
            "ID: 1, Title: Story 1, Intro: Intro 1, Tags: tag1, tag2"
            in complete_stories
        )
        assert (
            "ID: 2, Title: Story 2, Intro: Intro 2, Tags: tag3, tag4"
            in complete_stories
        )

    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test_key"})
    @patch("sekai_optimizer.nodes.generate_groundtruths.LangSmithClient")
    def test_generate_ground_truth_for_user_function(
        self, mock_langsmith_client, sample_story_data
    ):
        """Test the individual user ground truth generation function."""
        # Setup mocks
        mock_chain = Mock()
        mock_chain.invoke.return_value = {
            "recommendations": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }

        mock_runnable = Mock()
        mock_langsmith_client.return_value.pull_prompt.return_value = mock_runnable
        mock_runnable.__or__ = Mock(return_value=mock_chain)

        # Create a mock service
        mock_service = Mock()
        mock_service.story_data = sample_story_data

        # Test data
        user_profile = "choice-driven, high-agency, dominant protector strategist"
        complete_stories = "ID: 1, Title: Story 1..."

        # Execute
        result = _generate_ground_truth_for_user(
            user_profile, complete_stories, mock_service
        )

        # Verify
        assert len(result) == 10
        assert all(isinstance(rec, int) for rec in result)
        assert all(rec in sample_story_data for rec in result)

        # Verify LangSmith integration
        mock_langsmith_client.return_value.pull_prompt.assert_called_once_with(
            "generate_groundtruths:latest", include_model=True
        )
        mock_chain.invoke.assert_called_once()
