import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import faiss  # type: ignore
import numpy as np
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_and_save_index(
    client: openai.OpenAI,
    stories_data: List[Dict[str, Any]],
    index_filepath: Path,
    mapping_filepath: Path,
) -> None:
    """
    Builds a FAISS index from story data using OpenAI embeddings and saves it,
    along with an ID mapping.

    Args:
        client: An initialized OpenAI client.
        stories_data: A list of dictionaries, where each dict represents a story.
        index_filepath: Path to save the FAISS index file.
        mapping_filepath: Path to save the JSON mapping from index position to story ID.
    """
    texts_to_embed = [
        f"{story['title']}: {story.get('intro', '')}" for story in stories_data
    ]

    logger.info(f"Embedding {len(texts_to_embed)} stories using OpenAI...")

    # Get embeddings from OpenAI
    response = client.embeddings.create(
        input=texts_to_embed, model="text-embedding-3-small"
    )

    embeddings = [item.embedding for item in response.data]
    embeddings = np.array(embeddings, dtype="f4")  # type: ignore

    d = embeddings.shape[1]  # type: ignore

    # Build the FAISS index
    index = faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap(index)
    ids = np.arange(len(stories_data))
    index.add_with_ids(embeddings, ids)

    logger.info(f"Index built successfully with {index.ntotal} vectors.")

    # Save the index and mapping
    index_filepath.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_filepath))
    logger.info(f"FAISS index saved to {index_filepath}")

    id_mapping = {str(i): story["id"] for i, story in enumerate(stories_data)}
    with open(mapping_filepath, "w") as f:
        json.dump(id_mapping, f)
    logger.info(f"ID mapping saved to {mapping_filepath}")


if __name__ == "__main__":
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in .env file or environment variables."
        )

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "src" / "sekai_optimizer" / "data"

    stories_file = DATA_DIR / "stories.json"
    index_file = DATA_DIR / "stories.index"
    mapping_file = DATA_DIR / "stories_mapping.json"

    if not stories_file.exists():
        raise FileNotFoundError(
            f"'{stories_file}' not found. Please run synthesize_data.py first."
        )

    with open(stories_file, "r") as f:
        stories = json.load(f)

    openai_client = openai.OpenAI()

    build_and_save_index(openai_client, stories, index_file, mapping_file)
    logger.info("Index building complete.")
