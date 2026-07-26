# Formal E0-E3 experiments

This directory freezes and runs the formal KQA Pro and GrailQA experiments for
local Qwen (`qwen3-4b-16k`), version-pinned DeepInfra Qwen
(`Qwen/Qwen3-14B:01b0452aeedc42eaf4a73d87654a8129d6a436f0`) and OpenAI
(`gpt-5.4-2026-03-05`).

Formal sampling is fixed to:

```text
1-hop: 100
2-hop: 100
3-hop:  60
total: 260 questions per dataset
```

For GrailQA, a question is admitted only when the deterministic union of one
grounded proof path per Gold answer contains at most 50 statements. This keeps
every complete Gold proof theoretically retrievable under the fixed E3
`top_k=50` selector budget and excludes very broad answer-set questions.

Previously used development questions are excluded by default:

- the 15-question KQA Pro pilot;
- the 30-question GrailQA pilot.

This prevents smoke/pilot-driven prompt and pipeline decisions from leaking
into the formal evaluation.

On the current machine, use the working Python 3.12 interpreter:

```powershell
$py = "C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe"
```

Every model uses the same frozen questions. Candidate statement IDs, candidate
order, noise source and noise seed are shared across Document, Triple and
Graph. E2 always reads the saved E3 selections from the same model run; it
never launches a new selector.

GrailQA E1, E2 and E3 share the same persisted
`sufficient_gold_statement_ids`. E1 provides only this minimal grounded proof
union; E2 inserts any missing sufficient-Gold statements after reusing E3's
selection; E3 never forces Gold into the selected context.

At formal scale, selecting 60 executable 3-hop evaluation questions leaves
fewer than the pilot's requested 50 non-evaluation 3-hop background paths.
The formal KQA runner therefore preserves up to 50 paths per hop and then
deterministically fills the shared true-fact noise bank from remaining
non-evaluation paths in round-robin hop order. Each candidate bank still
contains exactly 100 statements and applies the same answer/distractor leakage
filters. The accepted path counts and final noise-bank size are audited in the
generated configuration.

## 1. Build KQA Pro inputs

No model or network call is used:

```powershell
& $py processing/formal_experiments/build_kqapro_formal_manifest.py
```

Output:

```text
processing/formal_experiments/output/inputs/kqapro_100_100_60/
```

## 2. Build GrailQA inputs

First validate that the dev set has enough eligible questions and that SPARQL
queries can be prepared. This does not contact Virtuoso:

```powershell
& $py processing/formal_experiments/build_grailqa_formal_inputs.py --dry-run
```

Then make sure the Freebase Virtuoso endpoint is running and materialize the
formal mini-KB:

```powershell
$env:GRAILQA_SPARQL_ENDPOINT = "http://localhost:3001/sparql"
& $py processing/formal_experiments/build_grailqa_formal_inputs.py
```

The materializer is resumable. It uses separate per-hop checkpoints because
the formal sample sizes are non-uniform.

Output:

```text
processing/formal_experiments/output/inputs/grailqa_100_100_60/
```

## 3. Review a frozen run plan

The command below validates the formal input and writes `run_plan.json`; it
does not call a model unless `--execute` is added.

KQA Pro with GPT-5.4:

```powershell
& $py processing/formal_experiments/run_formal_experiment.py `
  --dataset kqapro --backend openai
```

KQA Pro with Qwen:

```powershell
& $py processing/formal_experiments/run_formal_experiment.py `
  --dataset kqapro --backend ollama
```

GrailQA with GPT-5.4:

```powershell
& $py processing/formal_experiments/run_formal_experiment.py `
  --dataset grailqa --backend openai
```

GrailQA with Qwen:

```powershell
& $py processing/formal_experiments/run_formal_experiment.py `
  --dataset grailqa --backend ollama
```

Qwen3-14B through DeepInfra:

```powershell
$env:DEEPINFRA_TOKEN = "your-token"

& $py processing/formal_experiments/run_formal_experiment.py `
  --dataset kqapro --backend deepinfra

& $py processing/formal_experiments/run_formal_experiment.py `
  --dataset grailqa --backend deepinfra
```

These commands only validate inputs and write a plan. The token is read from
the environment and is never written to source code or experiment metadata.

### DeepInfra smoke inputs

Build a deterministic three-question manifest for each dataset, with one
question from each hop:

```powershell
& $py processing/formal_experiments/build_deepinfra_smoke_inputs.py
```

Smoke artifacts and runs are isolated under:

```text
processing/formal_experiments/output/smoke/deepinfra/
```

## 4. Execute

After inspecting `run_plan.json`, repeat a command with `--execute`. The runner
executes stages in the required order:

```text
E0 -> E1 -> E3 -> E2
```

Example:

```powershell
& $py processing/formal_experiments/run_formal_experiment.py `
  --dataset kqapro --backend openai --execute
```

For the complete DeepInfra formal run, use the checked launcher. It validates
the token and runs KQA Pro before GrailQA, stopping immediately if either
dataset fails:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File processing/formal_experiments/run_deepinfra_formal.ps1
```

Run its non-billable validation path first when needed:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File processing/formal_experiments/run_deepinfra_formal.ps1 -PlanOnly
```

To execute or validate only one dataset, add `-Dataset kqapro` or
`-Dataset grailqa`. The launcher reads `DEEPINFRA_TOKEN` from the current
process or the persistent user environment and never prints it.

Default outputs:

```text
processing/formal_experiments/output/runs/
  kqapro/openai/gpt-5.4-2026-03-05/{e0,e1,e3,e2}
  kqapro/ollama/qwen3-4b-16k/{e0,e1,e3,e2}
  grailqa/openai/gpt-5.4-2026-03-05/{e0,e1,e3,e2}
  grailqa/ollama/qwen3-4b-16k/{e0,e1,e3,e2}
  kqapro/deepinfra/Qwen_Qwen3-14B_01b045.../{e0,e1,e3,e2}
  grailqa/deepinfra/Qwen_Qwen3-14B_01b045.../{e0,e1,e3,e2}
```

Do not reuse an E3 selection file across models. OpenAI and Ollama runs must
remain in their separate output roots.
