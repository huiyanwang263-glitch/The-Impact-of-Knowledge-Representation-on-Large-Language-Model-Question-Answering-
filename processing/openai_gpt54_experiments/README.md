# GPT-5.4 experiments for KQA Pro and GrailQA

This directory runs the existing E0/E1/E2/E3 experiment logic through the
OpenAI Responses API. It does not modify the current Ollama scripts or their
outputs.

Default model: `gpt-5.4-2026-03-05` (fixed snapshot).

The adapter uses:

- `reasoning.effort = none`;
- `store = false`;
- Structured Outputs for GrailQA answer sets and both E3 selectors;
- no temperature or seed parameter;
- the same prompts, candidate banks, Gold forcing, evaluation and aliases as
  the existing experiments.

## Requirements

- Python 3.10 or newer; no third-party Python package is required;
- outbound HTTPS access to `https://api.openai.com`;
- an OpenAI API project, API key, billing/credits, and GPT-5.4 model access;
- the existing KQA Pro `kb.json`/`train.json` and frozen oracle manifest;
- the existing GrailQA selected questions, mini-KB, query results, source
  train/dev JSON files and answer-alias cache.

On this machine the working interpreter discovered during validation is:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe
```

The current `python` command and old `venv` point at an unavailable Windows
Store Python. Select Python 3.12 in the IDE, repair/recreate the virtual
environment, or replace `python` in the examples below with the full path
above.

## 1. API key

Do not paste the key into a Python file or commit it to Git. Set it in the
PowerShell session that will run the experiments:

```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

Optional configuration:

```powershell
$env:OPENAI_MODEL = "gpt-5.4-2026-03-05"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_REASONING_EFFORT = "none"
```

To use the cheaper mini model while keeping the same scripts:

```powershell
$env:OPENAI_MODEL = "gpt-5.4-mini"
```

The key must belong to an OpenAI API project with billing/credits enabled and
access to the requested model. ChatGPT Plus/Pro subscription access is separate
from API billing.

## 2. No-cost validation

These commands only build local inputs; they do not call OpenAI:

```powershell
python processing/openai_gpt54_experiments/run_kqapro_openai.py e0 --build-only
python processing/openai_gpt54_experiments/run_kqapro_openai.py e1 --build-only
python processing/openai_gpt54_experiments/run_kqapro_openai.py e3 --build-only

python processing/openai_gpt54_experiments/run_grailqa_openai.py e0 --build-only
python processing/openai_gpt54_experiments/run_grailqa_openai.py e1 --build-only
python processing/openai_gpt54_experiments/run_grailqa_openai.py e3 --build-only
```

E2 requires saved E3 selections, so its build-only validation should be done
after E3 selection has run.

## 3. Smoke tests

Run E3 before E2 because E2 reuses the same model's E3 selections.

GrailQA can directly select one question per hop:

```powershell
python processing/openai_gpt54_experiments/run_grailqa_openai.py e0 --max-per-hop 1
python processing/openai_gpt54_experiments/run_grailqa_openai.py e1 --max-per-hop 1
python processing/openai_gpt54_experiments/run_grailqa_openai.py e3 --max-per-hop 1
python processing/openai_gpt54_experiments/run_grailqa_openai.py e2 --max-per-hop 1
```

For KQA Pro, first create a three-question `oracle_questions.jsonl` containing
one frozen question for each hop, then pass the same file to every stage:

```powershell
python processing/openai_gpt54_experiments/run_kqapro_openai.py e0 --manifest path/to/kqa_smoke_3.jsonl
python processing/openai_gpt54_experiments/run_kqapro_openai.py e1 --manifest path/to/kqa_smoke_3.jsonl
python processing/openai_gpt54_experiments/run_kqapro_openai.py e3 --manifest path/to/kqa_smoke_3.jsonl
python processing/openai_gpt54_experiments/run_kqapro_openai.py e2 --manifest path/to/kqa_smoke_3.jsonl
```

The KQA wrapper dynamically validates the hop distribution in `--manifest`, so
it supports both the existing 15-question pilot and later `100/100/60` formal
manifests. It does not independently resample questions for any stage.

## 4. Existing 15/30-question bundles

KQA Pro:

```powershell
python processing/openai_gpt54_experiments/run_kqapro_openai.py e0 --manifest processing/knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl
python processing/openai_gpt54_experiments/run_kqapro_openai.py e1 --manifest processing/knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl
python processing/openai_gpt54_experiments/run_kqapro_openai.py e3 --manifest processing/knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl
python processing/openai_gpt54_experiments/run_kqapro_openai.py e2 --manifest processing/knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl
```

GrailQA:

```powershell
python processing/openai_gpt54_experiments/run_grailqa_openai.py e0
python processing/openai_gpt54_experiments/run_grailqa_openai.py e1
python processing/openai_gpt54_experiments/run_grailqa_openai.py e3
python processing/openai_gpt54_experiments/run_grailqa_openai.py e2
```

Outputs are isolated under:

```text
processing/openai_gpt54_experiments/output/kqapro/{e0,e1,e2,e3}
processing/openai_gpt54_experiments/output/grailqa/{e0,e1,e2,e3}
```

These generated directories are already covered by the repository's
`processing/**/output/` ignore rule.

## 5. Formal experiment inputs

Before a large run, replace the pilot manifests with the frozen formal
question files. Forward the original scripts' path options after the stage.
Examples:

```powershell
python processing/openai_gpt54_experiments/run_kqapro_openai.py e3 `
  --manifest path/to/kqapro_formal_questions.jsonl

python processing/openai_gpt54_experiments/run_grailqa_openai.py e3 `
  --questions path/to/grailqa_formal_questions.jsonl `
  --kb path/to/grailqa_formal_mini_kb.jsonl
```

Freeze and record all of the following before the formal run:

- exact model snapshot and API base;
- question manifest and candidate-bank SHA-256 hashes;
- prompt/source-code commit;
- reasoning effort (`none`);
- output-token limits and retry policy;
- E3 selection result hash reused by E2;
- API response token usage and any failed/retried request IDs.

The wrappers accept the original scripts' additional flags, such as
`--questions`, `--kb`, `--top-k`, `--max-per-hop`, `--selector-max-tokens` and
`--answer-max-tokens`.
