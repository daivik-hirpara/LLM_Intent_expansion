# Intent Expansion Pipeline: Approach & Learnings

## 1. Approach: The "Hybrid Scalable-Critic" Pipeline

We combined two powerful techniques to balance **Scalability** with **Reasoning Depth**:

### Phase 1: Scalable Data Reduction (Clustering)
*   **Problem**: Analyzing thousands of messages with an LLM is slow and expensive. Random sampling misses rare intents.
*   **Solution**: We use **Semantic Clustering** (SentenceTransformers + Agglomerative Clustering).
*   **Process**:
    1.  Embed all messages into vector space.
    2.  Cluster them to find semantic groups.
    3.  Select **Representative Samples** (top 3) from *each* cluster.
*   **Benefit**: This reduces the dataset from N messages to ~40-50 highly diverse samples that cover *every* pattern found in the data, not just the most common ones.

### Phase 2: The "Critic" Validation Loop (LLM)
*   **Problem**: LLMs can hallucinate or propose redundant intents.
*   **Solution**: A multi-step "Propose & Validate" workflow.
*   **Process**:
    1.  **Identify Issues**: The LLM analyzes the representative samples to find "Overloaded" or "Missing" intents.
    2.  **Validate Proposals**: A second LLM call acts as a "Critic", reviewing the proposals against the existing hierarchy to check for overlap, ambiguity, or unnecessary complexity.
*   **Benefit**: High-precision recommendations with significantly reduced noise.

## 2. Architecture

<img width="307" height="747" alt="image" src="https://github.com/user-attachments/assets/25b4195e-5bd2-46c4-b2a3-66baba0bd8fa" />


## 3. Findings & Justifications
Based on the provided dataset, the pipeline is designed to uncover splits such as:

### **New Secondary Intent: `product_safety_suitability`**
-   **Parent**: `specific_product`
-   **Reasoning**: Messages like "Age 10 year h koi side effects" and "Is it safe for oily scalp?" indicate a distinct concern about safety and suitability for specific demographics or conditions, which is different from general effectiveness or ingredients.
-   **Why Split?**: These are high-stakes queries often requiring medical disclaimers or specific safety assurances, distinct from general product info.

### **New Secondary Intent: `product_identification`**
-   **Parent**: `specific_product`
-   **Reasoning**: Users often simply state a product name (e.g., "Pigem ation cream") in response to "What are you looking for?".
-   **Why Split?**: This is a search/discovery intent, not a query *about* a product attribute. Identifying this allows the bot to trigger a product search flow immediately.

## 4. Scalability & Guardrails
-   **Scalability**: The pipeline scales with the number of *clusters*, not messages. Processing 10,000 messages might yield only 50 clusters, requiring only 50 LLM calls.
-   **Guardrails**:
    -   **Thresholding**: Clusters must meet a minimum size to be considered.
    -   **Context Awareness**: The embedding includes conversation history to resolve ambiguities (e.g., "Yes" -> "Yes to what?").
    -   **Deterministic Output**: The LLM is forced to output JSON, ensuring the pipeline doesn't break.

## 5. Limitations & Future Work
-   **Ambiguity**: Messages like "Okay" or "Thanks" often cluster together but are essentially noise (Basic Interactions). The pipeline handles this by mapping them to existing intents, but they can add noise to centroids.
-   **Multilingual Support**: The current embedding model is English-focused. For Hindi/Hinglish messages (present in data), a multilingual model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) would be better.
