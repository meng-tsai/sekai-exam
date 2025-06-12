"""
Recommend stories node implementation using Two-Stage RAG.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import faiss  # type: ignore
import numpy as np
import openai
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langsmith import Client as LangSmithClient
from pydantic import BaseModel

from ..data.state import OptimizationState

logger = logging.getLogger(__name__)


class StoryRecommendationsResponse(BaseModel):
    """Pydantic model for the expected LLM response format."""

    recommendations: List[int]


class RecommendationService:
    """Singleton service for handling story recommendations with Two-Stage RAG."""

    _instance = None

    def __init__(self):
        """Initialize the recommendation service with data loading."""
        self.faiss_index = None
        self.story_data = {}
        self.id_mapping = {}
        self.openai_client = None
        self._load_data()

    @classmethod
    def get_instance(cls):
        """Get singleton instance of RecommendationService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_data(self) -> None:
        """Load FAISS index, story data, and ID mapping into memory."""
        try:
            # Determine data directory path
            current_file = Path(__file__).resolve()
            data_dir = current_file.parents[1] / "data"

            # Load FAISS index
            index_path = data_dir / "stories.index"
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found at {index_path}")

            self.faiss_index = faiss.read_index(str(index_path))
            logger.debug(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

            # Load ID mapping
            mapping_path = data_dir / "stories_mapping.json"
            if not mapping_path.exists():
                raise FileNotFoundError(f"ID mapping not found at {mapping_path}")

            with open(mapping_path, "r") as f:
                self.id_mapping = json.load(f)
            logger.debug(f"Loaded ID mapping with {len(self.id_mapping)} entries")

            # Load story data
            stories_path = data_dir / "stories.json"
            if not stories_path.exists():
                raise FileNotFoundError(f"Stories data not found at {stories_path}")

            with open(stories_path, "r") as f:
                stories_list = json.load(f)

            # Convert to dict for O(1) lookup
            self.story_data = {story["id"]: story for story in stories_list}
            logger.debug(f"Loaded {len(self.story_data)} stories")

            # Initialize OpenAI client for embeddings
            self.openai_client = openai.OpenAI()
            logger.debug("Initialized OpenAI client for embeddings")

        except Exception as e:
            logger.error(f"Failed to load recommendation service data: {e}")
            raise

    def _get_user_embedding(self, user_tags: List[str]) -> np.ndarray:
        """Convert user tags to embedding vector."""
        if not user_tags:
            # For empty tags, use a generic query
            tags_text = "general story recommendations"
        else:
            # Join tags with commas
            tags_text = ", ".join(user_tags)

        try:
            response = self.openai_client.embeddings.create(
                input=[tags_text], model="text-embedding-3-small"
            )
            embedding = np.array(response.data[0].embedding, dtype="f4")
            return embedding.reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to get embedding for tags {user_tags}: {e}")
            raise

    def _faiss_retrieve(self, user_tags: List[str], k: int = 25) -> List[int]:
        """Stage 1: Retrieve top-k candidate stories using FAISS."""
        try:
            # Get embedding for user tags
            query_embedding = self._get_user_embedding(user_tags)

            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, k)

            # Convert FAISS indices to story IDs
            story_ids = []
            for idx in indices[0]:
                if str(idx) in self.id_mapping:
                    story_id = self.id_mapping[str(idx)]
                    story_ids.append(story_id)

            logger.debug(
                f"FAISS retrieved {len(story_ids)} candidates for tags: {user_tags}"
            )
            return story_ids

        except Exception as e:
            logger.error(f"FAISS retrieval failed for tags {user_tags}: {e}")
            raise

    def _llm_rerank(
        self, user_tags: List[str], candidate_stories: List[int], strategy_prompt: str
    ) -> List[int]:
        """Stage 2: Re-rank candidates using LLM to get top 10."""
        try:
            # Setup LangSmith client and chain
            langsmith_client = LangSmithClient(
                api_key=os.environ.get("LANGSMITH_API_KEY")
            )

            # Pull prompt with model from LangSmith Hub
            reranker_runnable = langsmith_client.pull_prompt(
                "recommend_stories_reranker:latest", include_model=True
            )

            # Setup JSON parser
            json_parser = JsonOutputParser(pydantic_object=StoryRecommendationsResponse)
            chain = reranker_runnable | json_parser

            # Format candidate stories for prompt
            candidate_stories_text = []
            for story_id in candidate_stories:
                if story_id in self.story_data:
                    story = self.story_data[story_id]
                    story_text = f"ID: {story_id}, Title: {story['title']}, Intro: {story.get('intro', '')}, Tags: {', '.join(story.get('tags', []))}"
                    candidate_stories_text.append(story_text)

            candidate_stories_formatted = "\n".join(candidate_stories_text)
            user_tags_formatted = (
                ", ".join(user_tags) if user_tags else "general interests"
            )

            # Call LLM
            recommendations = chain.invoke(
                {
                    "strategy_prompt": strategy_prompt,
                    "user_tags": user_tags_formatted,
                    "candidate_stories": candidate_stories_formatted,
                }
            )

            # Validate and ensure exactly 10 recommendations
            valid_recommendations = [
                rec
                for rec in recommendations
                if isinstance(rec, int) and rec in self.story_data
            ]

            # Pad with FAISS results if needed
            while len(valid_recommendations) < 10 and len(valid_recommendations) < len(
                candidate_stories
            ):
                for candidate in candidate_stories:
                    if candidate not in valid_recommendations:
                        valid_recommendations.append(candidate)
                        if len(valid_recommendations) >= 10:
                            break

            # Ensure exactly 10 recommendations
            final_recommendations = valid_recommendations[:10]

            logger.debug(
                f"LLM re-ranking returned {len(final_recommendations)} recommendations for tags: {user_tags}"
            )
            return final_recommendations

        except Exception as e:
            logger.error(f"LLM re-ranking failed for tags {user_tags}: {e}")
            # Fallback: return first 10 from FAISS results
            fallback_recommendations = candidate_stories[:10]
            logger.warning(
                f"Using fallback FAISS recommendations: {len(fallback_recommendations)} stories"
            )
            return fallback_recommendations

    def recommend_for_user(
        self, user_tags: List[str], strategy_prompt: str
    ) -> List[int]:
        """Generate 10 story recommendations for a single user using Two-Stage RAG."""
        # Stage 1: FAISS retrieval (25 candidates)
        candidate_stories = self._faiss_retrieve(user_tags, k=25)

        # Stage 2: LLM re-ranking (10 final)
        final_recommendations = self._llm_rerank(
            user_tags, candidate_stories, strategy_prompt
        )

        return final_recommendations

    def recommend_for_users(
        self, batch_tags: Dict[str, List[str]], strategy_prompt: str
    ) -> Dict[str, List[int]]:
        """Generate recommendations for multiple users using RunnableParallel approach."""
        try:
            # Create individual recommendation functions for each user
            user_runnables = {}
            for user_id, user_tags in batch_tags.items():
                user_runnables[user_id] = RunnableLambda(
                    lambda _, uid=user_id, tags=user_tags: self.recommend_for_user(
                        tags, strategy_prompt
                    )
                )

            # Execute in parallel
            parallel_runnable = RunnableParallel(user_runnables)
            results = parallel_runnable.invoke({})

            logger.debug(f"Generated recommendations for {len(results)} users")
            return results

        except Exception as e:
            logger.error(f"Batch recommendation failed: {e}")
            raise


def recommend_stories_node(state: OptimizationState) -> Dict[str, Any]:
    """
    Generate 10 story recommendations for each user using Two-Stage RAG.

    State I/O Contract:
    - Reads from State: state['batch_simulated_tags'], state['current_strategy_prompt']
    - Updates State with: { "batch_recommendations": Dict[str, List[int]] }
    """
    logger.info("=== RECOMMEND STORIES NODE ===")

    batch_tags = state["batch_simulated_tags"]
    strategy_prompt = state["current_strategy_prompt"]

    logger.debug(f"Processing recommendations for {len(batch_tags)} users")
    logger.debug(f"Strategy prompt: {strategy_prompt}")

    try:
        # Get recommendation service instance
        service = RecommendationService.get_instance()

        # Generate recommendations for all users
        batch_recommendations = service.recommend_for_users(batch_tags, strategy_prompt)

        # Log results
        for user_id, recommendations in batch_recommendations.items():
            logger.debug(f"User {user_id}: recommended {recommendations}")

        logger.debug(
            f"Completed story recommendations for {len(batch_recommendations)} users"
        )

        return {"batch_recommendations": batch_recommendations}

    except Exception as e:
        logger.error(f"Critical error in story recommendation: {e}", exc_info=True)
        # FAISS failure should hard fail the graph
        raise
