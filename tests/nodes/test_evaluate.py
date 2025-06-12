"""
Unit tests for evaluate_node implementation using TDD methodology.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from sekai_optimizer.nodes.evaluate import (
    evaluate_node,
    EvaluationService,
    FeedbackSynthesisResponse,
)
from sekai_optimizer.data.state import OptimizationState


class TestEvaluateNode:
    """Test suite for evaluate_node function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_state = {
            "current_user_batch": [
                {"user_id": 1, "profile": "User 1 profile"},
                {"user_id": 2, "profile": "User 2 profile"},
            ],
            "batch_recommendations": {
                "1": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "2": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
            },
            "batch_ground_truths": {
                "1": [101, 102, 103, 111, 112, 113, 114, 115, 116, 117],
                "2": [201, 202, 221, 222, 223, 224, 225, 226, 227, 228],
            },
            "current_strategy_prompt": "Test strategy prompt",
        }

        self.mock_stories = {
            101: {"id": 101, "title": "Story 101", "intro": "Intro 101"},
            102: {"id": 102, "title": "Story 102", "intro": "Intro 102"},
            103: {"id": 103, "title": "Story 103", "intro": "Intro 103"},
            201: {"id": 201, "title": "Story 201", "intro": "Intro 201"},
            202: {"id": 202, "title": "Story 202", "intro": "Intro 202"},
        }

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_normal_evaluation_flow(self, mock_service_instance):
        """Test normal evaluation flow with all metrics calculated successfully."""
        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        # Mock individual evaluation results
        mock_individual_results = {
            "1": {
                "user_id": "1",
                "precision_at_10": 0.3,
                "semantic_similarity": 0.75,
                "overlap_count": 3,
                "recommended_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                    {
                        "id": 102,
                        "title": "Story 102",
                        "intro": "Intro 102",
                        "tags": ["romance"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                    {
                        "id": 111,
                        "title": "Story 111",
                        "intro": "Intro 111",
                        "tags": ["mystery"],
                    },
                ],
            },
            "2": {
                "user_id": "2",
                "precision_at_10": 0.2,
                "semantic_similarity": 0.85,
                "overlap_count": 2,
                "recommended_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                    {
                        "id": 202,
                        "title": "Story 202",
                        "intro": "Intro 202",
                        "tags": ["sci-fi"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                    {
                        "id": 221,
                        "title": "Story 221",
                        "intro": "Intro 221",
                        "tags": ["horror"],
                    },
                ],
            },
        }

        mock_service.evaluate_user_batch.return_value = mock_individual_results
        mock_service.synthesize_feedback.return_value = "Test feedback message"

        # Execute
        result = evaluate_node(self.mock_state)

        # Assertions
        assert "evaluation_result" in result
        evaluation_result = result["evaluation_result"]

        assert evaluation_result["average_p10"] == 0.25  # (0.3 + 0.2) / 2
        assert (
            evaluation_result["average_semantic_similarity"] == 0.8
        )  # (0.75 + 0.85) / 2
        assert evaluation_result["synthesized_feedback"] == "Test feedback message"
        assert evaluation_result["batch_size"] == 2
        assert evaluation_result["failed_semantic_calculations"] == 0

        # Verify service calls
        mock_service.evaluate_user_batch.assert_called_once()
        mock_service.synthesize_feedback.assert_called_once()

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_perfect_match_scenario(self, mock_service_instance):
        """Test scenario with perfect matches between recommendations and ground truth."""
        # Setup identical recommendations and ground truth
        perfect_state = self.mock_state.copy()
        perfect_state["batch_recommendations"] = {
            "1": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "2": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
        }
        perfect_state["batch_ground_truths"] = {
            "1": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "2": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
        }

        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        mock_individual_results = {
            "1": {
                "user_id": "1",
                "precision_at_10": 1.0,
                "semantic_similarity": 1.0,
                "overlap_count": 10,
                "recommended_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                ],
            },
            "2": {
                "user_id": "2",
                "precision_at_10": 1.0,
                "semantic_similarity": 1.0,
                "overlap_count": 10,
                "recommended_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                ],
            },
        }

        mock_service.evaluate_user_batch.return_value = mock_individual_results
        mock_service.synthesize_feedback.return_value = "Perfect recommendations!"

        # Execute
        result = evaluate_node(perfect_state)

        # Assertions
        evaluation_result = result["evaluation_result"]
        assert evaluation_result["average_p10"] == 1.0
        assert evaluation_result["average_semantic_similarity"] == 1.0
        assert evaluation_result["synthesized_feedback"] == "Perfect recommendations!"

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_no_match_scenario(self, mock_service_instance):
        """Test scenario with no matches between recommendations and ground truth."""
        # Setup completely different recommendations and ground truth
        no_match_state = self.mock_state.copy()
        no_match_state["batch_recommendations"] = {
            "1": [301, 302, 303, 304, 305, 306, 307, 308, 309, 310],
            "2": [401, 402, 403, 404, 405, 406, 407, 408, 409, 410],
        }

        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        mock_individual_results = {
            "1": {
                "user_id": "1",
                "precision_at_10": 0.0,
                "semantic_similarity": 0.1,
                "overlap_count": 0,
                "recommended_stories": [
                    {
                        "id": 301,
                        "title": "Story 301",
                        "intro": "Intro 301",
                        "tags": ["action"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                ],
            },
            "2": {
                "user_id": "2",
                "precision_at_10": 0.0,
                "semantic_similarity": 0.05,
                "overlap_count": 0,
                "recommended_stories": [
                    {
                        "id": 401,
                        "title": "Story 401",
                        "intro": "Intro 401",
                        "tags": ["drama"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                ],
            },
        }

        mock_service.evaluate_user_batch.return_value = mock_individual_results
        mock_service.synthesize_feedback.return_value = (
            "Poor recommendations need improvement"
        )

        # Execute
        result = evaluate_node(no_match_state)

        # Assertions
        evaluation_result = result["evaluation_result"]
        assert evaluation_result["average_p10"] == 0.0
        assert evaluation_result["average_semantic_similarity"] == pytest.approx(
            0.075
        )  # (0.1 + 0.05) / 2

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_semantic_similarity_failure(self, mock_service_instance):
        """Test handling of semantic similarity calculation failures."""
        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        # Mock results with semantic similarity failures
        mock_individual_results = {
            "1": {
                "user_id": "1",
                "precision_at_10": 0.3,
                "semantic_similarity": None,  # Failed calculation
                "overlap_count": 3,
                "recommended_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                ],
            },
            "2": {
                "user_id": "2",
                "precision_at_10": 0.2,
                "semantic_similarity": None,  # Failed calculation
                "overlap_count": 2,
                "recommended_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                ],
            },
        }

        mock_service.evaluate_user_batch.return_value = mock_individual_results
        mock_service.synthesize_feedback.return_value = (
            "Feedback with failed semantic calculations"
        )

        # Execute
        result = evaluate_node(self.mock_state)

        # Assertions
        evaluation_result = result["evaluation_result"]
        assert evaluation_result["average_p10"] == 0.25
        assert (
            evaluation_result["average_semantic_similarity"] is None
        )  # No valid semantic scores
        assert evaluation_result["failed_semantic_calculations"] == 2
        # Check that the synthesize_feedback was called with None for semantic similarity
        call_args = mock_service.synthesize_feedback.call_args[0]
        assert call_args[3] is None  # average_semantic_similarity parameter

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_langsmith_synthesis_failure(self, mock_service_instance):
        """Test hard failure when LangSmith feedback synthesis fails."""
        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        mock_individual_results = {
            "1": {"user_id": "1", "precision_at_10": 0.3, "semantic_similarity": 0.75}
        }

        mock_service.evaluate_user_batch.return_value = mock_individual_results
        mock_service.synthesize_feedback.side_effect = Exception(
            "LangSmith connection failed"
        )

        # Execute and expect hard failure
        with pytest.raises(Exception, match="LangSmith connection failed"):
            evaluate_node(self.mock_state)

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_empty_batch_edge_case(self, mock_service_instance):
        """Test graceful handling of empty evaluation batch."""
        # Setup empty state
        empty_state = self.mock_state.copy()
        empty_state["current_user_batch"] = []
        empty_state["batch_recommendations"] = {}
        empty_state["batch_ground_truths"] = {}

        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        mock_service.evaluate_user_batch.return_value = {}
        mock_service.synthesize_feedback.return_value = "No users to evaluate"

        # Execute
        result = evaluate_node(empty_state)

        # Assertions
        evaluation_result = result["evaluation_result"]
        assert evaluation_result["average_p10"] == 0.0
        assert evaluation_result["average_semantic_similarity"] is None
        assert evaluation_result["batch_size"] == 0

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_partial_match_scenario(self, mock_service_instance):
        """Test scenario with partial overlap between recommendations and ground truth."""
        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        # Mock partial overlap results
        mock_individual_results = {
            "1": {
                "user_id": "1",
                "precision_at_10": 0.5,  # 5/10 overlap
                "semantic_similarity": 0.65,
                "overlap_count": 5,
                "recommended_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                    {
                        "id": 102,
                        "title": "Story 102",
                        "intro": "Intro 102",
                        "tags": ["romance"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 101,
                        "title": "Story 101",
                        "intro": "Intro 101",
                        "tags": ["adventure"],
                    },
                    {
                        "id": 111,
                        "title": "Story 111",
                        "intro": "Intro 111",
                        "tags": ["mystery"],
                    },
                ],
            },
            "2": {
                "user_id": "2",
                "precision_at_10": 0.3,  # 3/10 overlap
                "semantic_similarity": 0.55,
                "overlap_count": 3,
                "recommended_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                    {
                        "id": 202,
                        "title": "Story 202",
                        "intro": "Intro 202",
                        "tags": ["sci-fi"],
                    },
                ],
                "ground_truth_stories": [
                    {
                        "id": 201,
                        "title": "Story 201",
                        "intro": "Intro 201",
                        "tags": ["fantasy"],
                    },
                    {
                        "id": 221,
                        "title": "Story 221",
                        "intro": "Intro 221",
                        "tags": ["horror"],
                    },
                ],
            },
        }

        mock_service.evaluate_user_batch.return_value = mock_individual_results
        mock_service.synthesize_feedback.return_value = (
            "Moderate performance with room for improvement"
        )

        # Execute
        result = evaluate_node(self.mock_state)

        # Assertions
        evaluation_result = result["evaluation_result"]
        assert evaluation_result["average_p10"] == 0.4  # (0.5 + 0.3) / 2
        assert evaluation_result["average_semantic_similarity"] == pytest.approx(
            0.6
        )  # (0.65 + 0.55) / 2

    @patch("sekai_optimizer.nodes.evaluate.EvaluationService.get_instance")
    def test_evaluation_batch_failure(self, mock_service_instance):
        """Test hard failure when evaluation batch processing fails."""
        # Setup mock service
        mock_service = Mock()
        mock_service_instance.return_value = mock_service

        mock_service.evaluate_user_batch.side_effect = Exception(
            "Batch evaluation failed"
        )

        # Execute and expect hard failure
        with pytest.raises(Exception, match="Batch evaluation failed"):
            evaluate_node(self.mock_state)


class TestEvaluationService:
    """Test suite for EvaluationService class."""

    @pytest.fixture
    def mocked_evaluation_service(self):
        """Provides a fully mocked EvaluationService instance."""
        with patch("sekai_optimizer.nodes.evaluate.openai.OpenAI") as mock_openai_class:
            # Configure the mock class to return a specific mock instance
            mock_instance = MagicMock()
            mock_openai_class.return_value = mock_instance

            # Reset the singleton and create a new service for the test
            if EvaluationService._instance:
                EvaluationService._instance = None
            service = EvaluationService.get_instance()

            yield service

    def test_calculate_precision_at_10(self, mocked_evaluation_service):
        """Test P@10 calculation."""
        service = mocked_evaluation_service
        # Test perfect overlap
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ground_truth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert service.calculate_precision_at_10(recommended, ground_truth) == 1.0

        # Test partial overlap
        recommended = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
        ground_truth = [1, 2, 3, 6, 7, 8, 9, 10, 16, 17]
        assert (
            service.calculate_precision_at_10(recommended, ground_truth) == 0.3
        )  # 3/10

        # Test no overlap
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ground_truth = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        assert service.calculate_precision_at_10(recommended, ground_truth) == 0.0

    def test_format_story_for_embedding(self, mocked_evaluation_service):
        """Test story formatting for embedding."""
        service = mocked_evaluation_service

        story = {"title": "Test Story", "intro": "Test intro"}
        formatted = service._format_story_for_embedding(story)
        assert formatted == "Test Story: Test intro"

        # Test missing intro
        story = {"title": "Test Story"}
        formatted = service._format_story_for_embedding(story)
        assert formatted == "Test Story: "

    def test_embed_story_set(self, mocked_evaluation_service):
        """Test story set embedding."""
        service = mocked_evaluation_service
        # Setup mock OpenAI client
        mock_client = service.openai_client

        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response

        stories = [
            {"title": "Story 1", "intro": "Intro 1"},
            {"title": "Story 2", "intro": "Intro 2"},
        ]

        embedding = service._embed_story_set(stories)

        # Assertions
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args[1]
        assert "Story 1: Intro 1 | Story 2: Intro 2" in call_args["input"]

    def test_calculate_semantic_similarity(self, mocked_evaluation_service):
        """Test semantic similarity calculation."""
        service = mocked_evaluation_service
        mock_client = service.openai_client

        # Mock embedding responses for identical story sets
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [1.0, 0.0, 0.0]  # Unit vector
        mock_client.embeddings.create.return_value = mock_response

        stories1 = [{"title": "Story 1", "intro": "Intro 1"}]
        stories2 = [{"title": "Story 1", "intro": "Intro 1"}]

        similarity = service.calculate_semantic_similarity(stories1, stories2)

        # Should be 1.0 for identical embeddings
        assert similarity == 1.0

    def test_semantic_similarity_failure(self, mocked_evaluation_service):
        """Test semantic similarity calculation failure handling."""
        service = mocked_evaluation_service
        mock_client = service.openai_client
        mock_client.embeddings.create.side_effect = Exception("OpenAI API failed")

        stories1 = [{"title": "Story 1", "intro": "Intro 1"}]
        stories2 = [{"title": "Story 2", "intro": "Intro 2"}]

        similarity = service.calculate_semantic_similarity(stories1, stories2)

        # Should return None on failure
        assert similarity is None
