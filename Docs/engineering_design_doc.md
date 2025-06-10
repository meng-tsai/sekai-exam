# Sekai Agentic Recommendation Optimizer: Engineering Design Doc

**Version:** 2.2 (Final)
**Status:** Approved
**Author:** Meng-Huan Tsai

---

## 1. Overview

    This document outlines the engineering design for the Sekai Take-Home Challenge. The objective is to build an autonomous system—a "squad" of AI agents—that iteratively refines a recommendation prompt. The system's goal is to train a universal, high-quality prompt that enables a lightweight model to consistently recommend 10 highly-relevant stories to a diverse range of users.

    The proposed solution is an **agentic, self-optimizing loop** designed for robustness, scalability, and generalization. It leverages a three-component architecture (Evaluation Agent, Optimization Agent, Recommendation Workflow) and implements a sophisticated, multi-faceted strategy to prevent overfitting, ensuring the final prompt is effective for unseen users.

---

## 2. System Architecture: The Autonomous Optimization Loop

    The system is architected as a closed-loop workflow orchestrated around two core intelligent agents. Each component has a distinct role, and their interaction drives the system's learning process.

    ![Agent Loop Diagram](https://i.imgur.com/xI1Y6Bw.png)

    The core loop proceeds as follows:
    1.  **EVALUATE:** The `Evaluation Agent` assesses the current prompt's performance against a randomly selected user (or batch of users) from the training set.
    2.  **OPTIMIZE:** The `Prompt-Optimizer Agent` analyzes the evaluation score and qualitative feedback to propose a new, improved prompt.
    3.  **RECOMMEND:** The `Recommendation Workflow` uses this new prompt to generate recommendations, which are then fed back to the `Evaluation Agent` for the next cycle.

    This loop continues until a predefined stopping condition, based on validation set performance, is met.

---

## 3. Component Design & Responsibilities

### 3.1. Recommendation Component (Implemented as a Workflow)

    **Core Responsibility:** To generate 10 story recommendations for a user based on a set of input tags and a dynamic strategy prompt. This component is designed for **speed, quality, and scalability**.

    **Architecture: Two-Stage "Smart RAG" (Retrieval-Augmented Generation)**
    A production-ready two-stage architecture will be implemented to ensure scalability and semantic relevance.

    1.  **Stage 1: Retriever (Vector Search)**
        * **Embedding Model:** **OpenAI `text-embedding-3-small`**. This choice prioritizes state-of-the-art semantic understanding to ensure the highest quality candidates are retrieved.
        * **Vector Store:** **FAISS (Facebook AI Similarity Search)**. Chosen for its high performance and its ability to run entirely in-memory, making the project self-contained and easy to execute.
        * **Process:** Stories are pre-processed into vectors. At runtime, a user query is vectorized to retrieve the Top-K (e.g., K=25) most relevant candidate stories from the FAISS index.

    2.  **Stage 2: Ranker (LLM-based)**
        * **Technology:** A fast model like Gemini 2.0 Flash.
        * **Process:** The Ranker receives the K candidate stories and the `strategy_prompt`. It applies the strategy to this highly-relevant, reduced set to select and rank the final 10 stories.

    **I/O Contract:**
    * **Input:** `simulated_tags: List[str]`, `strategy_prompt: str`
    * **Output:** `recommended_story_ids: List[int]` (JSON format)

### 3.2. Evaluation Agent

    **Core Responsibility:** To act as the objective "judge," providing quantitative scores and qualitative feedback.

    **Internal Logic:**
    Designed as an agent with distinct tools (`tag_simulator`, `ground_truth_generator`, `scorer`). It uses a high-end reasoning model (e.g., Gemini 2.5 Pro) to simulate user behavior, establish a gold standard, and provide analytical feedback.

    **I/O Contract:**
    * **Input:** `user_profile: str`, `recommended_story_ids: List[int]`, `full_story_pool: List[Dict]`
    * **Output:** `evaluation_result: Dict` containing `score: float` (P@10), `semantic_score: float`, `feedback: str`, and `ground_truth_ids: List[int]`.

### 3.3. Prompt-Optimizer Agent

    **Core Responsibility:** To function as the "brain" of the operation, learning from past performance to evolve the recommendation strategy.

    **Agentic Design Rationale:**
    While its initial implementation relies on a single core "prompt refinement" tool, its design as an agent is a deliberate architectural choice. This framework provides **future extensibility** (the ability to add new tools) and enables **transparent reasoning** (through chain-of-thought logging), perfectly aligning with the challenge's request for an extensible, agent-driven system.

    **I/O Contract:**
    * **Input:** `previous_prompt: str`, `evaluation_result: Dict`, `evaluation_history: List[Dict]`
    * **Output:** `new_prompt: str`

---

## 4. Evaluation & Scoring Strategy

    A hybrid scoring approach will be used to capture both accuracy and semantic relevance.

    * **Primary Metric: Precision@10 (P@10)**
        * **Calculation:** `(Number of correctly recommended stories) / 10`
        * **Rationale:** Provides a clear, objective, and stable signal for optimization. It serves as the primary driver for the optimization loop and the core component of the stopping rule.

    * **Secondary Metric: Semantic Similarity Score**
        * **Calculation:** Cosine similarity between the averaged embedding vector of the recommended list and that of the ground truth list.
        * **Rationale:** This metric captures the "product feel." It rewards recommendations that are thematically aligned, even if they are not exact ID matches. It provides a more nuanced view of whether the prompt's "taste" is improving, which is crucial for a content recommendation product.

---

## 5. Comprehensive Generalization Strategy

    To combat overfitting and ensure the final prompt is robust, a multi-layered strategy will be implemented.

    **5.1. Data-Level Strategies**
    1.  **User Rotation:** Each optimization cycle will use a new, randomly selected user from the training set.
    2.  **Batch Evaluation:** To smooth out noise from outlier users, evaluation will be performed on a small batch (e.g., n=3) of users, with the average score used as the primary learning signal.
    3.  **Strategic User Synthesis:** The user pool will be expanded by generating diverse, sometimes oppositional, user archetypes to cover a wide spectrum of tastes.

    **5.2. Algorithm-Level Strategies**
    4.  **Explicit Generalization Instruction:** The `Prompt-Optimizer`'s meta-prompt will contain a critical instruction to create a *universal* prompt, not one tailored to the last tested user.
    5.  **Historical Context:** The `Prompt-Optimizer` will be provided with the last N (e.g., 5) evaluation cycles to identify trends and avoid oscillating between local optima.

    **5.3. Process-Level Strategies**
    6.  **Hold-Out Validation Set:** The synthesized user pool will be split into an 80% Training Set and a 20% Validation Set. The validation set is never used for training feedback.
    7.  **Early Stopping Rule:** The system's performance on the **validation set** will be checked periodically (e.g., every 5 cycles). The optimization loop will terminate when the validation score plateaus or begins to degrade for a sustained period (e.g., 3 consecutive checks). This is the definitive guard against overfitting.

---

## 6. Data & Caching Strategy

    * **Story & User Synthesis:** An LLM will generate ≈100 stories and ≈30-50 user profiles, split into 80% training and 20% validation sets.
    * **Embedding Index:** All story embeddings will be pre-computed and stored in the FAISS index.
    * **API Call Caching:** In a production environment, identical API calls to the LLM could be cached to reduce costs, though this is out of scope for the initial implementation.

---

## 7. Scalability & Production Readiness

    The choice of a **Two-Stage "Smart RAG"** architecture for the `Recommendation Component` is a deliberate decision to build a system that is scalable by design. This architecture ensures that the system can handle a corpus of 10,000+ stories with minimal changes, as the heavy lifting is handled by the efficient vector search retriever, while the expensive LLM Ranker operates on a small, relevant subset. This directly addresses the "how to scale to production volumes" requirement from the outset.

---

## 8. Technology Stack & Implementation Details

    * **Orchestration Framework:** **LangGraph** will be used to implement the stateful, cyclical workflow. Its graph-based paradigm is perfectly suited for managing the interactions and state transitions between the agents.
    * **Observability & Tracing:** **LangSmith** will be integrated for end-to-end tracing and debugging. This provides deep visibility into each agent's reasoning process and LLM calls, which is invaluable for analyzing performance and diagnosing issues. It will also be used to log the optimization cycles as required.
    * **Prompt Management:** **LangSmith Prompt Hub** will serve as the central repository for all system prompts. This enables version control, interactive testing in a playground environment, and seamless integration with the tracing system, greatly accelerating the prompt engineering lifecycle.
    * **Deployment & Reproducibility:** The entire application will be containerized using **Docker**. A `Dockerfile` and a simple `run.sh` script will be provided to ensure a true one-command execution, encapsulating all dependencies and guaranteeing reproducibility.
    * **Configuration Management:** Application configuration, including API keys (OpenAI, Gemini, LangSmith), and operational parameters (e.g., `MAX_ITERATIONS`, `VALIDATION_INTERVAL`), will be managed through a `.env` file. A `.env.example` will be provided for clarity, and the `.env` file will be included in `.gitignore`.
