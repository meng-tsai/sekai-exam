"""
Test suite for optimize_prompt_node implementation.
"""

import json
import os
from unittest.mock import Mock, patch

import pytest

from sekai_optimizer.data.state import OptimizationState
from sekai_optimizer.nodes.optimize_prompt import optimize_prompt_node


class TestOptimizePromptNode:
    """Test suite for optimize_prompt_node using TDD methodology."""

    def test_successful_prompt_optimization(self):
        """Test the primary success path where a new prompt is generated via LLM."""
        # Arrange
        mock_state = {
            "evaluation_result": {
                "average_p10": 0.7,
                "synthesized_feedback": "Users prefer more diverse recommendations with better genre matching.",
            },
            "current_strategy_prompt": "Focus on user preferences and story relevance.",
            "evaluation_history": [
                {"average_p10": 0.5, "synthesized_feedback": "Initial feedback"}
            ],
            "iteration_count": 1,
            "best_score": 0.5,
            "best_strategy_prompt": "Previous best prompt",
            "best_evaluation": {"average_p10": 0.5},
        }

        expected_new_prompt = "Enhanced strategy: Focus on user preferences, story relevance, and genre diversity for better recommendations."

        mock_chain_response = {"new_strategy_prompt": expected_new_prompt}

        # Mock LangSmith client and chain
        with (
            patch(
                "sekai_optimizer.nodes.optimize_prompt.LangSmithClient"
            ) as mock_langsmith_client,
            patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}),
        ):

            mock_client_instance = Mock()
            mock_langsmith_client.return_value = mock_client_instance

            mock_runnable = Mock()
            mock_client_instance.pull_prompt.return_value = mock_runnable

            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_chain_response

            with patch(
                "sekai_optimizer.nodes.optimize_prompt.JsonOutputParser"
            ) as mock_parser:
                mock_parser_instance = Mock()
                mock_parser.return_value = mock_parser_instance

                # Mock the chain combination (runnable | parser)
                mock_runnable.__or__ = Mock(return_value=mock_chain)

                # Act
                result = optimize_prompt_node(mock_state)

                # Assert
                mock_client_instance.pull_prompt.assert_called_once_with(
                    "optimize_recommendation:latest", include_model=True
                )

                # Verify chain was invoked with correct parameters
                mock_chain.invoke.assert_called_once()
                call_args = mock_chain.invoke.call_args[0][0]

                assert (
                    call_args["current_prompt"] == mock_state["current_strategy_prompt"]
                )
                assert call_args["current_score"] == 0.7
                assert (
                    call_args["current_feedback"]
                    == "Users prefer more diverse recommendations with better genre matching."
                )
                assert "Initial feedback" in call_args["evaluation_history"]

                # Verify returned state
                assert result["current_strategy_prompt"] == expected_new_prompt
                assert result["iteration_count"] == 2
                assert len(result["evaluation_history"]) == 2

    def test_new_best_score_is_tracked(self):
        """Test that the node correctly identifies and saves a new best-performing prompt and score."""
        # Arrange
        mock_state = {
            "evaluation_result": {
                "average_p10": 0.8,  # Higher than current best
                "synthesized_feedback": "Excellent performance improvement",
            },
            "current_strategy_prompt": "Current iteration prompt",
            "evaluation_history": [],
            "iteration_count": 2,
            "best_score": 0.5,  # Lower than current score
            "best_strategy_prompt": "Previous best prompt",
            "best_evaluation": {"average_p10": 0.5},
        }

        mock_chain_response = {"new_strategy_prompt": "Next iteration prompt"}

        # Mock LangSmith integration
        with (
            patch(
                "sekai_optimizer.nodes.optimize_prompt.LangSmithClient"
            ) as mock_langsmith_client,
            patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}),
        ):

            mock_client_instance = Mock()
            mock_langsmith_client.return_value = mock_client_instance

            mock_runnable = Mock()
            mock_client_instance.pull_prompt.return_value = mock_runnable

            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_chain_response

            with patch("sekai_optimizer.nodes.optimize_prompt.JsonOutputParser"):
                mock_runnable.__or__ = Mock(return_value=mock_chain)

                # Act
                result = optimize_prompt_node(mock_state)

                # Assert - Best results should be updated
                assert result["best_score"] == 0.8
                assert result["best_strategy_prompt"] == "Current iteration prompt"
                assert result["best_evaluation"]["average_p10"] == 0.8

    def test_score_does_not_beat_best_score(self):
        """Test that the node does not update best results if current score is lower."""
        # Arrange
        mock_state = {
            "evaluation_result": {
                "average_p10": 0.6,  # Lower than current best
                "synthesized_feedback": "Performance declined",
            },
            "current_strategy_prompt": "Current iteration prompt",
            "evaluation_history": [],
            "iteration_count": 3,
            "best_score": 0.8,  # Higher than current score
            "best_strategy_prompt": "Best performing prompt",
            "best_evaluation": {
                "average_p10": 0.8,
                "synthesized_feedback": "Best feedback",
            },
        }

        mock_chain_response = {"new_strategy_prompt": "Next iteration prompt"}

        # Mock LangSmith integration
        with (
            patch(
                "sekai_optimizer.nodes.optimize_prompt.LangSmithClient"
            ) as mock_langsmith_client,
            patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}),
        ):

            mock_client_instance = Mock()
            mock_langsmith_client.return_value = mock_client_instance

            mock_runnable = Mock()
            mock_client_instance.pull_prompt.return_value = mock_runnable

            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_chain_response

            with patch("sekai_optimizer.nodes.optimize_prompt.JsonOutputParser"):
                mock_runnable.__or__ = Mock(return_value=mock_chain)

                # Act
                result = optimize_prompt_node(mock_state)

                # Assert - Best results should remain unchanged
                assert result["best_score"] == 0.8
                assert result["best_strategy_prompt"] == "Best performing prompt"
                assert result["best_evaluation"]["average_p10"] == 0.8
                assert (
                    result["best_evaluation"]["synthesized_feedback"] == "Best feedback"
                )

    def test_history_and_iteration_are_updated(self):
        """Test that state management for history and iteration count is correct."""
        # Arrange
        initial_history = [
            {"average_p10": 0.3, "synthesized_feedback": "First iteration"},
            {"average_p10": 0.5, "synthesized_feedback": "Second iteration"},
        ]

        mock_state = {
            "evaluation_result": {
                "average_p10": 0.6,
                "synthesized_feedback": "Third iteration feedback",
            },
            "current_strategy_prompt": "Current prompt",
            "evaluation_history": initial_history,
            "iteration_count": 2,
            "best_score": 0.5,
            "best_strategy_prompt": "Best prompt",
            "best_evaluation": {"average_p10": 0.5},
        }

        mock_chain_response = {"new_strategy_prompt": "Fourth iteration prompt"}

        # Mock LangSmith integration
        with (
            patch(
                "sekai_optimizer.nodes.optimize_prompt.LangSmithClient"
            ) as mock_langsmith_client,
            patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}),
        ):

            mock_client_instance = Mock()
            mock_langsmith_client.return_value = mock_client_instance

            mock_runnable = Mock()
            mock_client_instance.pull_prompt.return_value = mock_runnable

            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_chain_response

            with patch("sekai_optimizer.nodes.optimize_prompt.JsonOutputParser"):
                mock_runnable.__or__ = Mock(return_value=mock_chain)

                # Act
                result = optimize_prompt_node(mock_state)

                # Assert
                assert result["iteration_count"] == 3  # Incremented from 2
                assert (
                    len(result["evaluation_history"]) == 3
                )  # Added current evaluation
                assert (
                    result["evaluation_history"][-1] == mock_state["evaluation_result"]
                )  # Latest evaluation added

    def test_langsmith_connection_failure_hard_fails(self):
        """Test that LangSmith connection failures cause the node to hard fail."""
        # Arrange
        mock_state = {
            "evaluation_result": {
                "average_p10": 0.7,
                "synthesized_feedback": "Test feedback",
            },
            "current_strategy_prompt": "Test prompt",
            "evaluation_history": [],
            "iteration_count": 1,
            "best_score": 0.5,
            "best_strategy_prompt": "Best prompt",
            "best_evaluation": {"average_p10": 0.5},
        }

        # Mock LangSmith client to raise an exception
        with (
            patch(
                "sekai_optimizer.nodes.optimize_prompt.LangSmithClient"
            ) as mock_langsmith_client,
            patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}),
        ):

            mock_langsmith_client.side_effect = Exception("LangSmith connection failed")

            # Act & Assert
            with pytest.raises(Exception, match="LangSmith connection failed"):
                optimize_prompt_node(mock_state)

    def test_llm_call_failure_hard_fails(self):
        """Test that LLM call failures cause the node to hard fail."""
        # Arrange
        mock_state = {
            "evaluation_result": {
                "average_p10": 0.7,
                "synthesized_feedback": "Test feedback",
            },
            "current_strategy_prompt": "Test prompt",
            "evaluation_history": [],
            "iteration_count": 1,
            "best_score": 0.5,
            "best_strategy_prompt": "Best prompt",
            "best_evaluation": {"average_p10": 0.5},
        }

        # Mock LangSmith client but make chain invocation fail
        with (
            patch(
                "sekai_optimizer.nodes.optimize_prompt.LangSmithClient"
            ) as mock_langsmith_client,
            patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}),
        ):

            mock_client_instance = Mock()
            mock_langsmith_client.return_value = mock_client_instance

            mock_runnable = Mock()
            mock_client_instance.pull_prompt.return_value = mock_runnable

            mock_chain = Mock()
            mock_chain.invoke.side_effect = Exception("LLM call failed")

            with patch("sekai_optimizer.nodes.optimize_prompt.JsonOutputParser"):
                mock_runnable.__or__ = Mock(return_value=mock_chain)

                # Act & Assert
                with pytest.raises(Exception, match="LLM call failed"):
                    optimize_prompt_node(mock_state)

    def test_pydantic_model_validation(self):
        """Test that the Pydantic model correctly validates LLM responses."""
        # This test verifies the Pydantic model structure
        from sekai_optimizer.nodes.optimize_prompt import OptimizedPromptResponse

        # Test valid response
        valid_response = {"new_strategy_prompt": "Test prompt"}
        model = OptimizedPromptResponse(**valid_response)
        assert model.new_strategy_prompt == "Test prompt"

        # Test invalid response should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            OptimizedPromptResponse(**{"invalid_field": "value"})
