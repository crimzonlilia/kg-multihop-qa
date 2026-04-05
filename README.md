# Knowledge Graph Multihop QA

This repository contains tools and scripts for building and evaluating knowledge graphs for multihop question answering (QA). The project supports various stages of the pipeline, including data extraction, graph construction, and evaluation.

---

## Features

- **Triple Extraction**: Extracts subject-relation-object triples from text passages.
- **Graph Construction**: Builds knowledge graphs from extracted triples.
- **Evaluation**: Evaluates the pipeline using metrics like AiC@k (Answer-in-Context).
- **Customizable**: Supports multiple embedding models and cache configurations.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/kg-multihop-qa.git
   cd kg-multihop-qa
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Build Knowledge Graph
Run the rebuild_graph.py script to extract triples and build a graph:
```bash
python rebuild_graph.py --samples 300 --cache 300
```

### 2. Evaluate Pipeline
Run the test_full_pipeline.py script to evaluate the pipeline:
```bash
python test_full_pipeline.py --samples 500 --cache 500 --embed-model bge-small
```

### 3. Inspect Triples
Use check_triples.py to inspect extracted triples:
```bash
python check_triples.py --cache 300
```

---

## Folder Structure

- src: Source code for extraction, graph building, and evaluation.
- data: Contains raw, processed, and cached data.
- reports: Stores evaluation results, logs, and benchmarks.
- scripts: Utility scripts for debugging and testing.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---
