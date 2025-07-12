# Context Engineering for Localized LLMs

**Author:** Partha Pratim Ray
**Date:** 12 July 2025
**Contact:** [parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com)

---

## Overview

This repository provides a **complete, research-grade context engineering pipeline for Localized Large Language Models (LLMs)** running on edge or resource-constrained devices. The code demonstrates semantic routing, memory management, dynamic context assembly, inference parameter selection, and logging, powered by local LLMs and embedding models (via Ollama) and designed for reproducible, real-world benchmarking and research.

* **Use cases:** Scientific research, AI agent benchmarking, context-aware chatbots, on-device LLM experiments, education.
* **Key features:** Dynamic context selection, short/long-term memory, semantic top-k retrieval, dynamic LLM config, cost modeling, reproducible logging.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Pipeline Architecture](#pipeline-architecture)
* [Requirements](#requirements)
* [Setup](#setup)
* [Usage](#usage)
* [Code Structure](#code-structure)
* [Configuration](#configuration)
* [Logging and Metrics](#logging-and-metrics)
* [Corpus and Memory Examples](#corpus-and-memory-examples)
* [Test Prompts](#test-prompts)
* [Citation](#citation)
* [License](#license)

---

## Features

* **Semantic Retrieval:** Context assembly via cosine similarity and local embedding models.
* **Short-term & Long-term Memory:** Simulates user recency and persistent profile.
* **Dynamic Context Engineering:** Modules for retrieval ($\Phi_R$), memory ($\Phi_M$), summarization ($\Phi_S$), ordering ($\Phi_O$), filtering ($\Phi_I$), and LLM config ($\Phi_G$).
* **Edge Optimized:** Designed to run with Ollama or any locally-hosted LLM/embedding model.
* **Full Logging:** All session parameters, metrics, context, and responses logged as a structured CSV for analysis.

---

## Pipeline Architecture

```
User Query
   │
   ▼
[R_phi_R]   (Top-k semantic retrieval)
   │
   ▼
[M_phi_M]   (Memory selection)
   │
   ▼
[S_phi_S]   (Summarization/compression)
   │
   ▼
[O_phi_O]   (Ordering/ranking)
   │
   ▼
[I_phi_I]   (Isolation/filtering)
   │
   ▼
[G_phi_G]   (LLM inference parameter selection)
   │
   ▼
Final Prompt → [LLM Inference]
   │
   ▼
Logging (CSV)
```

---

## Requirements

* **Python 3.8+**
* [Ollama](https://ollama.com/) running locally (for LLM + embedding API)
* Python packages:

  * `requests`
  * `numpy`
  * `csv`
  * `os`
  * `datetime`
  * `typing`
  * `time`
  * `random`

Install Python dependencies:

```bash
pip install numpy requests
```

---

## Setup

1. **Start Ollama** and make sure your target LLM (e.g., `qwen3:1.7b`) and embedding model (e.g., `nomic-embed-text`) are downloaded and running.
2. **Clone this repository** and navigate into it.
3. **Edit the configuration variables** at the top of the script if needed (model names, API URL, CSV path, etc).
4. **Prepare your own corpus/memory** (or use the examples provided).

---

## Usage

* **Run the script**:

  ```bash
  python context_engineering.py
  ```

* **Batch Mode:** The script will run all test prompts and log detailed results to `llm_log.csv`.

* **Custom usage:**
  Import `context_engineering_pipeline()` in your own project or adapt modules for research.

---

## Code Structure

* **Semantic Embedding Functions**

  * `get_embedding(text)`: Calls local embedding API for vector.
  * `cosine_similarity(vec_a, vec_b)`: Computes similarity.
  * `top_k_semantic(query, corpus, k)`: Top-k semantic retrieval.

* **Context Modules**

  * `R_phi_R(q, D)`: Retrieval module (dynamic top-k).
  * `M_phi_M(q, H_ST, H_LT)`: Memory selection (recency/history).
  * `S_phi_S(entries)`: Summarization/compression.
  * `O_phi_O(context_chunks, q)`: Ordering/ranking.
  * `I_phi_I(context_chunks, T)`: Filtering/deduplication.
  * `G_phi_G(q, C)`: Inference config selection (dynamic).

* **Cost Modeling**

  * `C_cost(...)`: Context cost calculation.
  * `G_cost(params)`: Generation cost calculation.

* **Logging**

  * `log_to_csv(...)`: Logs all pipeline metrics, context, response, and parameters for each query.

* **Pipeline**

  * `context_engineering_pipeline(...)`: Full end-to-end workflow.

---

## Configuration

At the top of the script, set:

* `OLLAMA_URL`: Base URL for Ollama API (default: `http://localhost:11434`)
* `LLM_MODEL`: Local LLM model name (e.g., `qwen3:1.7b`)
* `EMBED_MODEL`: Embedding model name (e.g., `nomic-embed-text`)
* `LOG_CSV_PATH`: Path to output CSV log

---

## Logging and Metrics

Every query logs the following:

* Timestamp
* User prompt & context
* LLM model name
* All selected inference parameters
* Ollama timing metrics (duration, tokens/sec, etc)
* Context cost, generation cost
* Micro-stage timings (summarization, ordering, filtering, config selection, all in ns)

This enables deep research on the effects of context engineering and LLM inference parameters on output, latency, and resource usage.

---

## Corpus and Memory Examples

* **Corpus:** Factual and domain knowledge (science, coding, history, pop culture, etc).
* **Short-Term Memory:** Simulates recent user queries, recency effects, and corrections.
* **Long-Term Memory:** Persistent user profile, interests, and preferences.

See `corpus`, `short_term_memory`, and `long_term_memory` definitions in the script for examples.

---

## Test Prompts

A set of diverse prompts is provided for benchmarking, covering:

* LLM context engineering theory
* Sampling, diversity, and temperature effects
* Science, history, coding, pop culture, meta-preference

You can add or edit these prompts as desired.

---

## Citation

If you use this codebase in your research, please cite as:

```
Partha Pratim Ray, "Context Engineering for Localized LLMs," GitHub repository, July 2025.
```

---

## License

MIT License.
Feel free to use, modify, and build upon this work for research or educational purposes.

---

## Contact

For questions or collaborations, contact Partha Pratim Ray at
[parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com).

---

**Happy research!**
If you find this useful, please star the repo and share feedback or improvements via pull request or issue.

---
