# Sekai Optimizer: Implementation Plan (v1.4)

**Document Version:** 1.4
**Architecture:** Granular Node Workflow
**Development Strategy:** Skeleton-First with TDD

---

## 1. Guiding Principles

    * **Skeleton-First Development:** The entire LangGraph application skeleton, including all nodes and edges, will be built first using mock/stub implementations. Real logic will be filled in incrementally.
    * **Test-Driven Development (TDD):** Each node's real logic will be developed against a pre-written test suite.
    * **Continuous Integration:** The E2E test will be established early, ensuring the system's integrity is maintained as each node is implemented.

---

## Stage 0: Project Foundation & Pre-flight Setup

    **Goal:** Establish a clean, reproducible project environment.

    * **Step 0.1: Initialize Project & Git**
        * Create project structure (`/nodes`, `/data`, `/tests`, `/scripts`, `/services`, `main.py`).
        * Initialize Git and configure `.gitignore`.
    * **Step 0.2: Environment & Tooling Setup**
        * Finalize `requirements.txt`.
        * Set up `Dockerfile` for one-command execution.
        * Create `.env.example` with `MAX_ITERATIONS`, `MAX_OPTIMIZATION_MINUTE`, `INIT_STRATEGY_PROMPT`, and `USER_PER_BATCH`.
        * Configure **LangSmith**.
    * **Step 0.3: Offline Data Preparation**
        * Implement and run a one-time script (`scripts/setup.py`) to:
            1. Synthesize all story and user data.
            2. Build and save the FAISS vector index using the chosen OpenAI embedding model.

---

## Stage 1: Graph Scaffolding & E2E Test Setup

    **Goal:** Build a fully runnable, end-to-end graph using placeholder logic to validate the overall architecture and data flow first.

    * **Step 1.1: Define Graph State**
        * In `main.py`, define the complete `StateGraph` TypedDict, including all fields necessary for the workflow (e.g., `start_time`, `iteration_count`, `current_user_batch`, `current_strategy_prompt`, `batch_recommendations`, etc.).
    * **Step 1.2: Implement Node Stubs**
        * For **every** node in the diagram, create a simple "stub" function in the `/nodes/` directory that accepts the `state` and returns a hardcoded, correctly formatted dictionary to update the state.
    * **Step 1.3: Assemble the Full Graph with Stubs**
        * In `main.py`, assemble the complete LangGraph application. Define all nodes and edges, including the conditional edge for the stopping logic.
    * **Step 1.4: Write the Initial E2E Test**
        * Create the `tests/test_e2e.py` script. This test will run the **fully stubbed graph** for a few iterations and assert that the state is passed and updated correctly, validating the architectural wiring.

---

## Stage 2: Incremental Node Implementation & Unit Testing (Detailed)

    **Goal:** Systematically replace each stub function with its real, tested implementation, one by one. After each node is implemented, the E2E test from Stage 1 must be run to ensure successful integration.

    * **Step 2.1: `pick_users_node`**
        * **Goal:** Select a random batch of users from the training set.
        * **State I/O Contract:**
            * **Reads from State:** `state['full_training_users']`, `state['config']['user_per_batch']`
            * **Updates State with:** `{ "current_user_batch": List[Dict] }`
        * **Implementation:** A pure Python function using `random.sample()`.
        * **TDD:** Write a unit test asserting the function returns a unique batch of the correct size from a mock user list.

    * **Step 2.2: `simulate_tags_node`**
        * **Goal:** For each user in the batch, simulate the tags they would select.
        * **State I/O Contract:**
            * **Reads from State:** `state['current_user_batch']`
            * **Updates State with:** `{ "batch_simulated_tags": Dict[str, List[str]] }`
        * **Implementation:** Iterates through the batch, calling a service method that uses a high-reasoning LLM with a prompt from LangSmith Hub.
        * **TDD:** Write a unit test for a single mock user. Mock the LLM call. Assert the node returns a correctly formatted dictionary.

    * **Step 2.3: `recommend_stories_node`**
        * **Goal:** Generate 10 story recommendations for each user.
        * **State I/O Contract:**
            * **Reads from State:** `state['batch_simulated_tags']`, `state['current_strategy_prompt']`
            * **Updates State with:** `{ "batch_recommendations": Dict[str, List[int]] }`
        * **Implementation:** The node calls a service encapsulating the two-stage RAG logic (FAISS retrieval + LLM re-ranking).
        * **TDD:** Write an integration test using a mock FAISS index and mock the LLM Ranker call. Assert the output is a dictionary mapping user IDs to lists of 10 story IDs.

    * **Step 2.4: `generate_groundtruths_node`**
        * **Goal:** Generate the "gold-standard" recommendations for each user.
        * **State I/O Contract:**
            * **Reads from State:** `state['current_user_batch']`
            * **Updates State with:** `{ "batch_ground_truths": Dict[str, List[int]] }`
        * **Implementation:** Iterates through the batch, calling a service method that uses a high-reasoning LLM.
        * **TDD:** Write a unit test for a single mock user. Mock the LLM call. Assert the output dictionary is correctly formatted.

    * **Step 2.5: `evaluate_node`**
        * **Goal:** To produce a single, generalizable evaluation report for the entire batch.
        * **State I/O Contract:**
            * **Reads from State:** `state['batch_recommendations']`, `state['batch_ground_truths']`, `state['current_user_batch']`
            * **Updates State with:** `{ "evaluation_result": Dict }`
        * **Implementation:** Implements the two-stage feedback mechanism: first, computes individual scores, then calls a service to synthesize a single, high-level feedback string.
        * **TDD:** Write a unit test providing mock data. Mock the final synthesis LLM call. Assert the node returns a single dictionary containing `average_p10` and `synthesized_feedback`.

    * **Step 2.6: `optimize_prompt_node`**
        * **Goal:** Generate the next-generation strategy prompt.
        * **State I/O Contract:**
            * **Reads from State:** `state['evaluation_result']`, `state['current_strategy_prompt']`, `state['evaluation_history']`
            * **Updates State with:** `{ "current_strategy_prompt": str, "evaluation_history": List[Dict], "iteration_count": int }` (updates prompt, appends to history, increments counter)
        * **Implementation:** Calls a service that pulls the meta-prompt from LangSmith Hub, formats it, and calls the reasoning LLM.
        * **TDD:** Write a unit test providing a mock `evaluation_result`. Mock the LLM call. Assert the node returns a new, non-empty prompt string.

---

## Stage 3: Final Validation & Tuning

    **Goal:** Run the fully implemented system and prepare for submission.

    * **Step 3.1: Full System Run**
        * Execute the `main.py` script via the Docker one-command `run.sh`.
        * Monitor the entire process using the **LangSmith** traces.
    * **Step 3.2: Analyze Results**
        * Review the LangSmith dashboard to extract the table of optimization cycles.

---

## Stage 4: Finalization & Deliverables

    **Goal:** Package the project for submission.

    * **Step 4.1: Finalize `README.md`**
        * Document the final architecture, development approach, and setup/run instructions.
    * **Step 4.2: Record Demo & Submit**
        * Record the video, clean up the code, and submit the repository.
