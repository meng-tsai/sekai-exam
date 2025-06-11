# Sekai Optimizer: Implementation Plan (v1.2)

**Document Version:** 1.2
**Architecture:** Granular Node Workflow based on User Diagram
**Based on Design Doc:** `v2.4 (Final, Clarified)`

---

## 1. Guiding Principles

    * **Test-Driven Development (TDD):** Every node in the graph will have its own dedicated test written *before* implementation.
    * **Modular Node Implementation:** Each white box in the diagram will be implemented as an independent, testable Python function (a LangGraph node).
    * **Incremental Graph Assembly:** We will build and test nodes individually, then assemble them into the final graph, ensuring reliability at each step.

---

## Stage 0: Project Foundation & Pre-flight Setup

    **Goal:** Establish a clean, reproducible, and professional project structure.

    * **Step 0.1: Initialize Project & Git**
        * Create the main project directory and initialize Git.
        * Establish the directory structure: `/nodes` (for node logic), `/data`, `/tests`, `/scripts`.
        * Create `.gitignore` and `.env.example` files.

    * **Step 0.2: Environment & Tooling Setup**
        * Finalize `requirements.txt` with all necessary packages.
        * Set up the `Dockerfile` for one-command execution.
        * Configure **LangSmith** for observability.

---

## Stage 1: Foundational Data & Offline Nodes

    **Goal:** Prepare all necessary data assets before the main optimization loop begins. These nodes correspond to the pre-loop setup.

    * **Step 1.1: `synthesize_data_node`**
        * **Goal:** Generate the story and user datasets.
        * **Test:** Write a test to assert that the script generates the correct number and format of stories/users and splits them correctly into training/validation sets.
        * **Implement:** Create a script in `/scripts/setup.py` that encapsulates this logic. This will be run once during the Docker build or via a setup command.

    * **Step 1.2: `build_index_node`**
        * **Goal:** Create and save the FAISS vector index.
        * **Test:** Write a test to assert that the script loads stories, uses the OpenAI embedding model, and saves a valid FAISS index file to disk.
        * **Implement:** Add this logic to the `/scripts/setup.py` script to be run after data synthesis.

---

## Stage 2: Core Loop Node Implementation & Unit Testing

    **Goal:** Implement and individually test each node that will be part of the main LangGraph loop.

    * **Step 2.1: `pick_users_node`**
        * **Goal:** Randomly select a batch of users from the training set.
        * **Test:** Write a unit test (`tests/test_nodes.py`) that provides a mock user list and asserts the function returns a batch of the correct size and that all users are from the provided list.
        * **Implement:** Create the function in `/nodes/data_nodes.py`.

    * **Step 2.2: `simulate_tags_node`**
        * **Goal:** Simulate tags for each user in the batch.
        * **Test:** Write a unit test that mocks the LLM call. It will provide a sample user profile and assert that the function returns a list of strings.
        * **Implement:** Create the function in `/nodes/evaluation_nodes.py`.

    * **Step 2.3: `recommend_stories_node`**
        * **Goal:** Execute the RAG workflow for each user in the batch.
        * **Test:** Write an integration test using a mock FAISS index. It will assert that for a given user's tags and a strategy prompt, the function returns exactly 10 story IDs.
        * **Implement:** Create the function in `/nodes/recommendation_nodes.py`.

    * **Step 2.4: `generate_groundtruths_node`**
        * **Goal:** Generate the ground truth list for each user in the batch.
        * **Test:** Write a unit test that mocks the LLM call. Provide a user profile and assert it returns a list of 10 integer IDs.
        * **Implement:** Create the function in `/nodes/evaluation_nodes.py`.

    * **Step 2.5: `evaluate_node`**
        * **Goal:** Calculate scores and synthesize a single, generalizable feedback string for the entire batch.
        * **Test:** Write a unit test providing mock recommendations and ground truths for a batch. It should assert the function returns a dictionary with the correct average scores and a non-empty feedback string. Mock the LLM call used for synthesis.
        * **Implement:** Create the function in `/nodes/evaluation_nodes.py`, implementing the two-stage feedback mechanism.

    * **Step 2.6: `optimize_prompt_node`**
        * **Goal:** Generate a new, improved strategy prompt.
        * **Test:** Write a unit test that mocks the LLM call. Provide a sample evaluation result and assert the function returns a new prompt string.
        * **Implement:** Create the function in `/nodes/optimizer_nodes.py`.

---

## Stage 3: Graph Assembly, Prompt Management & E2E Testing

    **Goal:** Wire all tested nodes together, ensure robust prompt management, and verify the entire system functions as a cohesive whole.

    * **Step 3.1: Centralize Prompts in LangSmith Hub**
        * **Goal:** To establish a single source of truth for all system prompts, enabling versioning and rapid, code-free iteration.
        * **Action:** Create the initial versions of the system's meta-prompts (for the evaluator and optimizer) directly in the **LangSmith Prompt Hub**.
        * **Principle:** It is a strict requirement that **no prompts are hardcoded** or stored in local files within the codebase. All prompts **must** be pulled from the LangSmith Hub at runtime (e.g., using `langchain.hub.pull()`). This decouples prompt engineering from application logic.

    * **Step 3.2: Define Graph State and Assemble Graph**
        * **Implement:** In `main.py`, define the `StateGraph`'s state (e.g., current prompt, user batch, evaluation result, etc.).
        * **Implement:** Add each function from the `/nodes/` directory as a node to the graph, ensuring they pull their respective prompts from the Hub.
        * **Implement:** Define the edges connecting the nodes in the precise sequence from the diagram.

    * **Step 3.3: Implement Conditional Edge (`determine_stop_node`)**
        * **Implement:** Create the conditional logic that checks the stopping condition. This edge will either loop back to the `pick_users_node` or proceed to the `END` of the graph.

    * **Step 3.4: Final End-to-End (E2E) Test**
        * **Test:** Create a comprehensive E2E test script (`tests/test_e2e.py`) that runs the compiled LangGraph app for 2-3 full loops using a small, fixed dataset and real API calls.
        * **Test:** It will assert that the application state is correctly updated at each step and that the process completes without errors.

---

## Stage 4: Finalization & Deliverables

    **Goal:** Prepare the project for submission.

    * **Step 4.1: Finalize `README.md`**
        * Document the final, granular architecture, explaining the role of each node and the prompt management strategy via LangSmith Hub.
        * Provide clear setup and execution instructions.

    * **Step 4.2: Record Demo & Submit**
        * Record a short video demonstrating the one-command execution and walking through the LangSmith traces to show the graph's execution and prompt lineage.
        * Clean up the code, add final comments, and submit the repository.
