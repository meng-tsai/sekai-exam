# Sekai Optimizer: Implementation Plan

**Document Version:** 1.0
**Based on Design Doc:** `v2.2 (Final)`

---

## 1. Guiding Principles

    * **Test-Driven Development (TDD):** For every piece of logic, a corresponding test will be written *first*. Implementation follows the test's specification. This ensures correctness and facilitates refactoring.
    * **Phased Development:** The project is broken down into logical stages. Each stage delivers a testable, functional component, allowing for incremental progress and early validation.
    * **Comprehensive Testing:** The plan includes unit tests for individual functions, integration tests for components, and a final end-to-end (E2E) test for the entire system.

---

## Stage 0: Project Foundation & Environment Setup

    **Goal:** Establish a clean, reproducible, and professional project structure.

    * **Step 0.1: Initialize Project Structure**
        * Create the main project directory.
        * Initialize a Git repository (`git init`).
        * Create the core directory structure (`/agents`, `/workflows`, `/data`, `/tests`).
        * Create an initial `.gitignore` file, ensuring to include `.env`, `__pycache__/`, and other common Python artifacts.

    * **Step 0.2: Dependency & Environment Management**
        * Create `requirements.txt` and add initial core dependencies (e.g., `langchain`, `langgraph`, `openai`, `google-generativeai`, `python-dotenv`, `faiss-cpu`, `sentence-transformers`).
        * Create the `Dockerfile` and a `docker-compose.yml` (optional, but recommended) for easy execution.
        * Create the `.env.example` file listing all required environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `LANGCHAIN_API_KEY`, etc.).
        * Create the local `.env` file for development.

    * **Step 0.3: Configure Tooling**
        * Set up a new project in **LangSmith** to obtain the necessary credentials.
        * Verify that environment variables are correctly loaded and that basic connections to LangSmith can be established.

---

## Stage 1: Data Pipeline & Core Recommendation Workflow

    **Goal:** Build and test the data foundation and the core "action" component of the system.

    * **Step 1.1: Data Synthesis & Preparation**
        * **Test:** Write a test (`tests/test_data.py`) to verify that the data synthesis script generates the correct number of stories and users, and that the output data structure matches the expected schema (e.g., a list of dicts with required keys).
        * **Implement:** Create a Python script (`scripts/synthesize_data.py`) that calls an LLM to generate ≈100 stories and ≈50 user profiles, saving them as JSON files in the `/data` directory.

    * **Step 1.2: Embedding and Vector Store Creation (Retriever)**
        * **Test:** Write a test to ensure the embedding script correctly loads the story data, uses the OpenAI embedding model, and successfully creates and saves a FAISS index file. The test should assert that the number of vectors in the index matches the number of stories.
        * **Implement:** Create a script (`scripts/build_index.py`) that performs the one-time embedding process and saves the FAISS index to disk.

    * **Step 1.3: Recommendation Workflow Implementation (Ranker)**
        * **Test:** Write an integration test (`tests/test_recommendation.py`) for the two-stage RAG workflow. This test will use a pre-built, small mock FAISS index. It will assert that given a set of tags, the workflow function correctly queries the index, calls the ranker LLM (mocked), and returns a list of exactly 10 story IDs in the specified JSON format.
        * **Implement:** Create the `RecommendationWorkflow` in `/workflows/recommendation.py`. This function will encapsulate the logic for vectorizing a query, retrieving candidates from FAISS, and then using the Ranker LLM to produce the final list.

---

## Stage 2: The "Judge" - Evaluation Agent

    **Goal:** Build and test the component responsible for scoring the recommendations.

    * **Step 2.1: Implement Scoring Logic**
        * **Test:** Write pure unit tests (`tests/test_scoring.py`) for the metric calculation functions. For `calculate_p10`, provide known lists and assert the correct precision. For `calculate_semantic_similarity`, provide known vectors and assert the correct cosine similarity.
        * **Implement:** Create the scoring utility functions in a `/utils/scoring.py` file.

    * **Step 2.2: Implement Evaluation Tools**
        * **Test:** Write unit tests for the agent's internal tools (`ground_truth_generator`, `tag_simulator`). These tests will mock the LLM calls and assert that the tools return data in the correct format (e.g., a list of 10 integers for ground truth).
        * **Implement:** Create the tool functions that will be used by the Evaluation Agent.

    * **Step 2.3: Assemble and Test the Evaluation Agent**
        * **Test:** Write an integration test (`tests/test_evaluation_agent.py`) for the complete `EvaluationAgent`. This test will provide a mock user profile and a list of recommended IDs. It will assert that the agent correctly calls its tools (mocked) and returns a final `evaluation_result` dictionary containing all the required keys (`score`, `feedback`, etc.) with the correct data types.
        * **Implement:** In `/agents/evaluation.py`, assemble the tools and the agent logic using the LangChain agent framework.

---

## Stage 3: The "Brain" - Prompt-Optimizer Agent

    **Goal:** Build and test the component responsible for evolving the prompt.

    * **Step 3.1: Set up Prompt Hub**
        * Create the initial versions of the system's meta-prompts (for the evaluator and optimizer) directly in the **LangSmith Prompt Hub**.

    * **Step 3.2: Implement and Test the Prompt-Optimizer Agent**
        * **Test:** Write an integration test (`tests/test_optimizer_agent.py`). Provide it with a mock `evaluation_result` and `evaluation_history`. The test will mock the LLM call and assert that the agent returns a non-empty string, which represents the new prompt.
        * **Implement:** In `/agents/optimizer.py`, create the `PromptOptimizerAgent`. Its logic will pull the relevant meta-prompt from the LangSmith Hub, combine it with the inputs, and call the reasoning model to generate the next strategy prompt.

---

## Stage 4: End-to-End System Integration & Testing

    **Goal:** Assemble all tested components into the final LangGraph application and verify the entire loop functions correctly.

    * **Step 4.1: Define the State and Graph**
        * **Implement:** In `main.py`, define the application's state graph in **LangGraph**. This involves defining the nodes (each agent/workflow) and the edges that dictate the flow of information (e.g., from evaluator to optimizer). Define the logic for the entry point and the conditional edges (e.g., checking the stopping rule).

    * **Step 4.2: Final End-to-End (E2E) Test**
        * **Test:** Create a comprehensive E2E test script (`tests/test_e2e.py`). This script will:
            1. Use a small, fixed subset of the data (e.g., 10 stories, 5 users).
            2. Run the entire LangGraph application for a fixed number of iterations (e.g., 2 cycles).
            3. The test will **not** mock any LLM calls but will use real API calls to ensure full integration.
            4. It will assert that the application completes without errors and that the final state contains an optimized prompt. This test verifies that all components are wired together correctly and can communicate as designed.
        * **Refine:** Debug and refine the LangGraph implementation based on the E2E test results.

---

## Stage 5: Finalization & Deliverables

    **Goal:** Prepare the project for submission.

    * **Step 5.1: Finalize Documentation**
        * Write a comprehensive `README.md` explaining the project, architecture, how to set it up (using `.env.example`), and how to run it using the Docker command.
        * Include a section showing the final, optimized prompt and a sample table of the optimization cycles logged from LangSmith.

    * **Step 5.2: Record Demo Video**
        * Record a short (≤ 5 minutes) video walking through the project's architecture, code, and a live run of the one-command demo.

    * **Step 5.3: Code Cleanup & Submission**
        * Remove any debugging print statements, ensure all code is well-commented, and perform a final check of the repository before submission.
