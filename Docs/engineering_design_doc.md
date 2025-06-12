# Sekai Agentic Recommendation Optimizer: Engineering Design Doc

**Version:** 2.5 (Final)
**Status:** Approved
**Author:** Meng-Huan Tsai

---

## 1. Overview

    This document outlines the engineering design for the Sekai Take-Home Challenge. The objective is to build an autonomous system that iteratively refines a recommendation prompt. The system's goal is to train a universal, high-quality prompt that enables a lightweight model to consistently recommend 10 highly-relevant Sekai stories to a diverse range of users.

    The proposed solution is a **structured, self-optimizing workflow** designed for robustness, scalability, and generalization. It is implemented as a stateful graph where each step is a distinct, testable node.

---

## 2. System Architecture: The Granular Optimization Loop

    The system is architected as a closed-loop, stateful graph using LangGraph. Each node represents a discrete step in the process, ensuring a deterministic and controllable flow.

    The core loop proceeds through the following sequence of nodes:

    1.  **`pick_users`**: A batch of users is randomly selected from the training set.
    2.  **`simulate_tags`**: Tags are simulated for each user in the batch.
    3.  **`recommend_stories`**: Recommendations are generated using the current strategy prompt.
    4.  **`generate_groundtruths`**: "Gold-standard" recommendations are generated.
    5.  **`evaluate`**: The recommendations are scored and a generalizable feedback report is synthesized.
    6.  **`optimize_prompt`**: A new, improved strategy prompt is generated based on the feedback.
    7.  **`determine_stop` (Conditional Edge)**: The loop either repeats with a new batch or terminates.

    This architecture directly translates the user-provided diagram into a robust engineering plan.

---

## 3. Node Design & Responsibilities

    The system is composed of independent nodes that communicate via a shared `State` object. Heavy, static resources (like the FAISS index and embedding models) are initialized once at startup and accessed via dedicated service classes, not passed through the state.

### 3.1. Recommendation Component (as a `recommend_stories` Node)

    **Core Responsibility:** To generate 10 story recommendations.
    **Architecture: Two-Stage "Smart RAG" (Retrieval-Augmented Generation)**

    1.  **Stage 1: Retriever (Vector Search)**
        * **Embedding Model:** **OpenAI `text-embedding-3-small`**.
        * **Vector Store:** **FAISS**.
        * **Process:** Retrieves Top-K candidate stories based on semantic similarity to user tags.

    2.  **Stage 2: Ranker (LLM-based)**
        * **Technology:** A fast model like Gemini 2.0 Flash.
        * **Process:** Ranks the K candidates based on the `strategy_prompt` and user tags.

### 3.2. Evaluation Component (as an `evaluate` Agentic Node)

    **Core Responsibility:** To provide a single, generalizable feedback report for a batch.
    **Agentic Rationale:** This node is termed "agentic" because it encapsulates a complex, multi-step reasoning process. It autonomously orchestrates a suite of tools (`tag_simulator`, `ground_truth_generator`, `scorer`, `feedback_synthesizer`) to achieve its goal, all while being a predictable node within the larger workflow.

### 3.3. Optimizer Component (as an `optimize_prompt` Agentic Node)

    **Core Responsibility:** To evolve the recommendation strategy prompt.
    **Agentic Rationale:** This node is "agentic" as it leverages an LLM's reasoning, guided by a sophisticated meta-prompt that considers historical context to generate an improved prompt. Its internal complexity is self-contained, while its role in the graph remains deterministic.

---

## 4. Evaluation & Scoring Strategy

    A hybrid scoring approach will be used:

    * **Primary Metric: Precision@10 (P@10):** Provides a clear, objective optimization signal.
    * **Secondary Metric: Semantic Similarity Score:** Captures thematic alignment, providing a smoother gradient for optimization.

---

## 5. Comprehensive Generalization Strategy

    A multi-layered strategy will be implemented to combat overfitting:

    * **Data-Level:** User Rotation, Batch Evaluation, and Strategic User Synthesis.
    * **Algorithm-Level:** Explicit Generalization Instruction in the Optimizer's meta-prompt and providing Historical Context.
    * **Process-Level:** A strict **Hold-Out Validation Set** and an **Early Stopping** mechanism based on validation set performance.

---

## 6. Data & Caching Strategy

    * **Story & User Synthesis:** An LLM will generate ≈100 stories and ≈30-50 user profiles.
    * **Embedding Index:** All story embeddings will be pre-computed and stored in the FAISS index.
    * **API Call Caching:** Out of scope for this challenge.

---

## 7. Scalability & Production Readiness

    The **Two-Stage "Smart RAG"** architecture is inherently scalable, addressing the production volume requirement from the outset by separating efficient retrieval from sophisticated ranking.

---

## 8. Technology Stack & Implementation Details

    * **Core Framework: LangChain & LangGraph**
        * **LangChain** provides the foundational abstractions and integrations (model wrappers, prompt templates).
        * **LangGraph** is used to define and execute the stateful, cyclical workflow. It provides the **explicit, controllable structure** for the system, ensuring predictable execution flow between nodes. The "agentic" behavior is encapsulated *within* specific nodes, not in the overall control flow, which remains deterministic.

    * **Observability & Tracing: LangSmith**
        * Will be integrated for end-to-end tracing and debugging.

    * **Prompt Management: LangSmith Prompt Hub**
        * Will serve as the central repository for all system prompts. It is a strict requirement that **no prompts are hardcoded**; they must be pulled from the Hub at runtime to enable rapid, code-free iteration.

    * **Deployment & Reproducibility: Docker**
        * The entire application will be containerized to ensure a true one-command execution.

    * **Configuration Management:**
        * API keys and operational parameters will be managed through a `.env` file.
        * Key configuration variables: `MAX_ITERATIONS`, `MAX_OPTIMIZATION_MINUTE`, `INIT_STRATEGY_PROMPT`, `USER_PER_BATCH`.
