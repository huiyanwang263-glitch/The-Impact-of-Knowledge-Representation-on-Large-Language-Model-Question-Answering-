# Knowledge Representation Experiments for KQA Pro and 2WikiMultihopQA

This repository contains data-processing and evaluation pipelines for comparing
knowledge representations and retrieval settings. The main controlled KB
experiments use KQA Pro; supplementary document-grounded experiments use
2WikiMultihopQA.

The dataset files themselves are not redistributed by this repository.

## Processing pipelines

### KQA Pro

- `closed_book_processing`: closed-book baseline experiments.
- `knowledge_representation_processing`: canonical statement generation and
  oracle-context experiments.
- `noisy_oracle_processing`: controlled gold-plus-noise experiments.
- `rag_full_kb_processing`: full-KB and atomic-KB corpus construction and RAG.
- `e2_forced_gold_processing`: forced-gold evidence experiments using retrieval
  results from the RAG pipeline.

### 2WikiMultihopQA

- `twowiki_closed_book_processing`: D0 closed-book baseline on fixed
  compositional/inference questions.
- `twowiki_gold_only_processing`: D1 gold-only representation experiment using
  2Wiki evidence triples and supporting sentences.

Each directory contains its own README with commands and experiment-specific
details.

## KQA Pro dataset setup

Obtain KQA Pro from its official repository:

<https://github.com/shijx12/KQAPro_Baselines>

Place the required dataset files in the project root:

```text
final-project/
|-- kb.json
|-- train.json
|-- test.json
|-- val.json
`-- processing/
```

The dataset files are intentionally excluded from Git. Do not commit them to
this repository. Follow the upstream KQA Pro license and citation requirements
when obtaining and using the data.

Most scripts locate `kb.json` and `train.json` in the project root by default.
Where available, the `--kb` and `--questions` arguments can be used to supply
different paths.

## Installation

Create and activate a Python virtual environment, then install the project
dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Model-backed experiments also require an OpenAI-compatible endpoint. Configure
it through the environment variables used by the individual pipeline scripts:

```powershell
$env:OPENAI_BASE_URL = "http://localhost:11434/v1"
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "your-model"
```

## Generated outputs

Pipeline outputs are written beneath each processing directory's `output/`
folder. These generated corpora, retrieval results, predictions, and caches are
excluded from Git and can be reproduced by following the corresponding README.

## Experiment design

See [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for the experimental rationale,
conditions, controls, and evaluation plan.

2Wiki experiments are designed as a document-grounded supplement to the KQA Pro
controlled KB setting. For representation-controlled comparisons, use 2Wiki
`evidences` as the canonical fact layer and render aligned document-style,
triple, and graph variants from the same facts.
