import pytest
import numpy as np
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import faiss  # type: ignore

# Assume the function will be in this module
from seikai_optimizer.scripts import build_index

# Mock data that resembles stories.json
MOCK_STORIES_DATA = [
    {
        "id": 1,
        "title": "Story One",
        "tags": ["a", "b"],
        "intro": "Intro for story one.",
    },
    {
        "id": 2,
        "title": "Story Two",
        "tags": ["c", "d"],
        "intro": "Intro for story two.",
    },
    {
        "id": 3,
        "title": "Story Three",
        "tags": ["e", "f"],
        "intro": "Intro for story three.",
    },
]

MOCK_EMBEDDINGS = np.array(
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype="f4"
)


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client for embedding calls."""
    with patch("openai.OpenAI") as mock_client_class:
        mock_instance = MagicMock()

        # Mock the response from the embeddings API
        mock_embedding_response = MagicMock()
        embedding_objects = [
            MagicMock(embedding=vec.tolist()) for vec in MOCK_EMBEDDINGS
        ]
        mock_embedding_response.data = embedding_objects

        mock_instance.embeddings.create.return_value = mock_embedding_response
        mock_client_class.return_value = mock_instance
        yield mock_instance


def test_build_and_save_index_with_openai(mock_openai_client, tmp_path):
    """
    Tests the main index building function using a mocked OpenAI client.
    - Verifies that the function calls the OpenAI embedding API correctly.
    - Verifies that it creates and saves a FAISS index and a mapping file.
    """
    index_path = tmp_path / "stories.index"
    mapping_path = tmp_path / "stories_mapping.json"

    # Call the function to be tested
    build_index.build_and_save_index(
        client=mock_openai_client,
        stories_data=MOCK_STORIES_DATA,
        index_filepath=index_path,
        mapping_filepath=mapping_path,
    )

    # 1. Assert OpenAI embedding model was called with the correct texts
    expected_texts = [f"{s['title']}: {s.get('intro', '')}" for s in MOCK_STORIES_DATA]
    mock_openai_client.embeddings.create.assert_called_once_with(
        input=expected_texts, model="text-embedding-3-small"
    )

    # 2. Assert that the FAISS index file was created
    assert index_path.exists()

    # 3. Assert the index contains the correct number of vectors
    loaded_index = faiss.read_index(str(index_path))
    assert loaded_index.ntotal == len(MOCK_STORIES_DATA)
    # The dimension should match the mock embeddings' dimension
    assert loaded_index.d == MOCK_EMBEDDINGS.shape[1]

    # 4. Assert that the ID mapping file was created and is correct
    assert mapping_path.exists()
    with open(mapping_path, "r") as f:
        id_mapping = json.load(f)

    expected_mapping = {
        str(i): story["id"] for i, story in enumerate(MOCK_STORIES_DATA)
    }
    assert id_mapping == expected_mapping

    print(
        "\nTest passed: `build_and_save_index` correctly used OpenAI embeddings and saved the FAISS index and ID mapping."
    )
