"""
Tests for simulate_tags node.
"""

from unittest.mock import patch, MagicMock

from sekai_optimizer.nodes.simulate_tags import simulate_tags_node, UserTagsResponse
from sekai_optimizer.data.state import OptimizationState


class TestSimulateTagsNode:
    """Test suite for simulate_tags_node."""

    def setup_method(self):
        """Setup test data."""
        self.sample_users = [
            {
                "user_id": 1,
                "profile": "Loves action-packed anime with strong protagonists and epic battles",
            },
            {
                "user_id": 2,
                "profile": "Prefers romantic comedies and slice-of-life stories with school settings",
            },
            {
                "user_id": 3,
                "profile": "Enjoys dark psychological thrillers and supernatural mystery stories",
            },
        ]

        self.state = OptimizationState(
            current_user_batch=self.sample_users, config={"user_per_batch": 3}
        )

    @patch("sekai_optimizer.nodes.simulate_tags.LangSmithClient")
    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key"})
    def test_successful_tag_simulation(self, mock_langsmith_client):
        """Test successful tag simulation with valid LLM response."""
        # Mock LangSmith client and chain
        mock_client = MagicMock()
        mock_langsmith_client.return_value = mock_client

        mock_runnable = MagicMock()
        mock_client.pull_prompt.return_value = mock_runnable

        # Mock LLM response
        mock_llm_response = {
            "user_tags": {
                "1": [
                    "action",
                    "epic-fantasy",
                    "strong-protagonist",
                    "battles",
                    "adventure",
                ],
                "2": [
                    "romance",
                    "comedy",
                    "slice-of-life",
                    "school-life",
                    "light-hearted",
                ],
                "3": ["psychological", "thriller", "supernatural", "mystery", "dark"],
            }
        }

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response
        mock_runnable.__or__.return_value = mock_chain

        # Execute
        result = simulate_tags_node(self.state)

        # Assertions
        assert "batch_simulated_tags" in result
        batch_tags = result["batch_simulated_tags"]

        # Should have all users
        assert "1" in batch_tags
        assert "2" in batch_tags
        assert "3" in batch_tags

        # Each user should have tags
        assert isinstance(batch_tags["1"], list)
        assert isinstance(batch_tags["2"], list)
        assert isinstance(batch_tags["3"], list)

        # Check specific content
        assert batch_tags["1"] == [
            "action",
            "epic-fantasy",
            "strong-protagonist",
            "battles",
            "adventure",
        ]
        assert batch_tags["2"] == [
            "romance",
            "comedy",
            "slice-of-life",
            "school-life",
            "light-hearted",
        ]
        assert batch_tags["3"] == [
            "psychological",
            "thriller",
            "supernatural",
            "mystery",
            "dark",
        ]

    @patch("sekai_optimizer.nodes.simulate_tags.LangSmithClient")
    def test_langsmith_connection_failure(self, mock_langsmith_client):
        """Test handling when LangSmith client fails."""
        # Mock LangSmith client to raise exception
        mock_langsmith_client.side_effect = Exception("LangSmith connection failed")

        # Execute
        result = simulate_tags_node(self.state)

        # Should return empty tags for all users
        expected_fallback = {"1": [], "2": [], "3": []}
        assert result["batch_simulated_tags"] == expected_fallback

    def test_empty_user_batch(self):
        """Test handling empty user batch."""
        empty_state = OptimizationState(
            current_user_batch=[], config={"user_per_batch": 0}
        )

        result = simulate_tags_node(empty_state)

        assert result["batch_simulated_tags"] == {}

    def test_user_tags_response_pydantic_model(self):
        """Test the Pydantic model validation."""
        # Valid data
        valid_data = {
            "user_tags": {"1": ["action", "adventure"], "2": ["romance", "comedy"]}
        }

        model = UserTagsResponse(**valid_data)
        assert model.user_tags["1"] == ["action", "adventure"]
        assert model.user_tags["2"] == ["romance", "comedy"]
