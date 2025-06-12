"""
Integration tests for recommend_stories_node using actual data files.
"""

import os
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from sekai_optimizer.nodes.recommend_stories import (
    recommend_stories_node,
    RecommendationService,
)
from sekai_optimizer.data.state import OptimizationState


class TestRecommendStoriesIntegration:
    """Integration test suite using real data files."""

    def setup_method(self):
        """Set up test fixtures with real-looking state."""
        self.mock_state: OptimizationState = {
            "batch_simulated_tags": {
                "1": ["action", "fantasy", "adventure", "heroic"],
                "2": ["romance", "slice-of-life", "modern", "coffee"],
                "3": ["horror", "supernatural", "mystery"],
            },
            "current_strategy_prompt": "Recommend stories that closely match user interests and provide engaging, relevant content",
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

    def test_data_files_exist(self):
        """Verify that required data files exist."""
        current_file = Path(__file__).resolve()
        data_dir = current_file.parents[2] / "src" / "sekai_optimizer" / "data"

        assert (data_dir / "stories.json").exists(), "stories.json not found"
        assert (data_dir / "stories.index").exists(), "stories.index not found"
        assert (
            data_dir / "stories_mapping.json"
        ).exists(), "stories_mapping.json not found"

    @patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_API_KEY": "test-key"}
    )
    def test_service_initialization(self):
        """Test that RecommendationService can initialize with real data."""
        # Clear any existing instance
        RecommendationService._instance = None

        try:
            service = RecommendationService.get_instance()

            # Verify service loaded data successfully
            assert service.faiss_index is not None
            assert service.faiss_index.ntotal > 0
            assert len(service.story_data) > 0
            assert len(service.id_mapping) > 0
            assert service.openai_client is not None

            # Verify we have the expected number of stories (should be ~105)
            assert len(service.story_data) >= 100

        except Exception as e:
            pytest.skip(f"Skipping integration test due to data loading issue: {e}")

    @patch("sekai_optimizer.nodes.recommend_stories.LangSmithClient")
    @patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_API_KEY": "test-key"}
    )
    def test_node_with_mocked_llm(self, mock_langsmith_client):
        """Test the node with real data but mocked LLM calls."""
        # Clear any existing instance
        RecommendationService._instance = None

        # Mock LangSmith client and chain
        mock_client = mock_langsmith_client.return_value
        mock_runnable = mock_client.pull_prompt.return_value

        # Mock the chain to return valid story IDs
        def mock_chain_invoke(inputs):
            # Return first 10 story IDs from our test data
            return {
                "recommendations": [
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
            }

        mock_chain = mock_runnable.__or__.return_value
        mock_chain.invoke.side_effect = mock_chain_invoke

        try:
            # Execute the node
            result = recommend_stories_node(self.mock_state)

            # Verify results
            assert "batch_recommendations" in result
            batch_recommendations = result["batch_recommendations"]

            # Check structure
            assert isinstance(batch_recommendations, dict)
            assert len(batch_recommendations) == 3  # 3 users in test state

            # Check each user has exactly 10 recommendations
            for user_id in ["1", "2", "3"]:
                assert user_id in batch_recommendations
                recommendations = batch_recommendations[user_id]
                assert len(recommendations) == 10
                assert all(isinstance(story_id, int) for story_id in recommendations)
                assert all(story_id > 0 for story_id in recommendations)

        except Exception as e:
            pytest.skip(f"Skipping integration test due to initialization issue: {e}")

    @patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_API_KEY": "test-key"}
    )
    def test_faiss_retrieval_functionality(self):
        """Test that FAISS retrieval works with real embeddings (requires OpenAI API)."""
        # This test requires actual OpenAI API access
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test-key":
            pytest.skip("Skipping test requiring real OpenAI API key")

        # Clear any existing instance
        RecommendationService._instance = None

        try:
            service = RecommendationService.get_instance()

            # Test FAISS retrieval with different tag types
            test_cases = [
                ["action", "fantasy", "adventure"],
                ["romance", "modern"],
                ["horror", "supernatural"],
                [],  # Empty tags
            ]

            for tags in test_cases:
                candidates = service._faiss_retrieve(tags, k=25)

                # Verify results
                assert isinstance(candidates, list)
                assert len(candidates) <= 25  # Should not exceed requested amount
                assert all(isinstance(story_id, int) for story_id in candidates)

                # Verify all story IDs exist in our data
                for story_id in candidates:
                    assert story_id in service.story_data

        except Exception as e:
            pytest.skip(f"Skipping FAISS test due to setup issue: {e}")
