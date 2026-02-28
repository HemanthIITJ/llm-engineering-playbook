# Large Language Models (LLMs) Cheat Book

Welcome to the **LLMs Cheat Book**, a rigorous and technically deep collection of notes covering everything from the fundamental building blocks of modern NLP to the absolute state-of-the-art in Generative AI. 

Whether you are a researcher, AI engineer, or student, this guide provides structured, mathematically accurate, and visually comprehensive explanations of the core concepts in the field.

---

## 📖 Table of Contents & Reading Order

For the best learning experience, it is recommended to read the chapters in the following order:

### Part 1: Foundations
*   **[1. Word Embeddings](./Embeddings.md)**
    *   *Covers:* Distributional hypothesis, vector semantics, count-based vs. prediction-based models (Word2Vec, SGNS), and embedding bias.
*   **[2. Language Models](./LMs.md)**
    *   *Covers:* Autoregressive factorization, n-grams to neural LMs, early RNNs, scaling laws (Kaplan, Chinchilla), and emergent capabilities.

### Part 2: Architecture & Pretraining
*   **[3. Transformers](./Transformers.md)**
    *   *Covers:* The complete Transformer architecture, scaled dot-product attention, multi-head self-attention, positional encodings (RoPE, ALiBi), and efficient attention variants (FlashAttention).
*   **[4. Language Model Pretraining](./Pretraining.md)**
    *   *Covers:* Self-supervised objectives (AR, MLM, Span Corruption), ELMo, encoder-based models (BERT, RoBERTa, ELECTRA), and evaluation benchmarks (GLUE, SuperGLUE).

### Part 3: Adaptation & Alignment
*   **[5. Fine-Tuning and Alignment](./FT_Alignment.md)**
    *   *Covers:* Supervised Fine-Tuning (SFT), Instruction Tuning, and Alignment techniques including RLHF (PPO) and Direct Preference Optimization (DPO).
*   **[6. Parameter-Efficient Fine-Tuning (PEFT)](./PEFT.md)**
    *   *Covers:* Model compression, Knowledge Distillation, Quantization (GPTQ, AWQ, QLoRA), and structured adaptation methods like LoRA, DoRA, and Adapters.

### Part 4: Advanced Paradigms
*   **[7. Prompt Engineering](./Prompting.md)**
    *   *Covers:* In-context learning, Chain-of-Thought (CoT), Tree of Thoughts (ToT), few-shot prompting strategies, and prompt optimization.
*   **[8. Augmented Language Models (ALMs)](./ALMs.md)**
    *   *Covers:* Retrieval-Augmented Generation (RAG) architecture, Vector Indices, Tool Calling, and Agentic frameworks (ReAct, Self-RAG).
*   **[9. Multilingual & Multimodal LLMs](./Multimodal.md)**
    *   *Covers:* Cross-lingual transfer, tokenization equity, contrastive alignment (CLIP), early vs. late fusion, and multimodal instruction tuning.
*   **[10. Recent Advances](./Recent.md)**
    *   *Covers:* The latest frontier developments, novel architectures (e.g., State Space Models/Mamba, Mixture of Experts), and advanced inference optimization techniques.

---

### 🎨 Visual Assets
This directory also contains an `assets/` folder with generated, scientifically accurate 8K diagrams and architectural visualisations embedded within the markdown files.

---

**Happy Learning!**
