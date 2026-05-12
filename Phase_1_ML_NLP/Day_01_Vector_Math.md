# Day 1: Demystifying Vector Math & Embeddings

**Date:** May 12, 2026
**Focus:** Understanding how AI represents language mathematically and how Vector Databases retrieve information.

## 🧠 Core Concept: What is an Embedding?
LLMs and RAG pipelines do not "read" English. They process data by translating concepts into numerical arrays (vectors) within a high-dimensional space. Modern models (like OpenAI's `text-embedding-3-small`) use thousands of dimensions to capture semantic meaning.

## 📐 The Math: Cosine Similarity
To find the most relevant information in a Vector Database (like ChromaDB or FAISS), the system calculates the **Cosine Similarity** between the user's prompt vector and the document vectors. It measures the *angle* between vectors rather than their magnitude.

**Formula:**
`similarity = cos(θ) = (A · B) / (||A|| * ||B||)`
* **1.0**: Semantic match (Pointing same direction)
* **0.0**: Unrelated (Perpendicular)
* **-1.0**: Antonyms (Opposite direction)

## 💻 Python Implementation (From Scratch)
Instead of relying on LangChain abstractions, I built the cosine similarity calculation from scratch using `numpy` and a local Hugging Face embedding model (`all-MiniLM-L6-v2`).

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load an open-source embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define text samples
text_a = "The child is exhibiting signs of severe anxiety."
text_b = "The minor shows symptoms of extreme nervousness."
text_c = "The financial report indicates a drop in quarterly revenue."

# 3. Generate embeddings (384-dimensional vectors)
vec_a = model.encode(text_a)
vec_b = model.encode(text_b)
vec_c = model.encode(text_c)

# 4. Cosine Similarity Function
def calculate_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    return dot_product / (magnitude_v1 * magnitude_v2)

# 5. Execution & Results
print(f"Similarity (A and B - Semantic Match): {calculate_cosine_similarity(vec_a, vec_b):.4f}")
print(f"Similarity (A and C - Unrelated): {calculate_cosine_similarity(vec_a, vec_c):.4f}")
