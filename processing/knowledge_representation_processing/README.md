# KQA Pro knowledge-representation pilot

This folder builds three representations from one canonical subset of
`kb.json`:

1. `documents`: natural-language documents grouped by subject;
2. `unordered_triples`: the same statements in a deterministic random order;
3. `readable_graph`: the same statements as labelled nodes and edges.

The representations are aligned by `statement_id`. The canonical layer keeps
qualifiers, literal types, and units exactly; each rendered form preserves
their semantic content. This avoids changing the knowledge content while
changing its presentation.

## Run

From the workspace root:

```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\build_pilot_representations.py
```

The copy of `kb.json` inside `kqa-pro-datasets` may be a Git LFS pointer. The
script detects that case and uses the real workspace-level `kb.json` by
default.

The default pilot starts from `Q513809` (`Mighty Morphin Power Rangers`), uses
a deterministic breadth-first expansion, selects at most 15 connected
entities, retains at most four attributes per entity, and uses random seed 42
for triple shuffling. Entities with more than 100 relations are retained as
leaf nodes but are not expanded; this prevents hubs such as countries from
swamping a small topical pilot.

Parameters can be changed without editing the script:

```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\build_pilot_representations.py `
  --seed-id Q786 `
  --max-entities 20 `
  --max-attributes-per-entity 3 `
  --shuffle-seed 42 `
  --output-dir processing\knowledge_representation_processing\output\q786
```

## Output

The default output directory is `output/pilot/`.

| File | Purpose |
| --- | --- |
| `manifest.json` | Source, seed, parameters, counts, and alignment checksum |
| `canonical_statements.jsonl` | Single source of truth for all representations |
| `documents.jsonl` | Machine-readable natural-language documents |
| `documents.md` | Human-readable document view |
| `unordered_triples.jsonl` | Machine-readable shuffled triples |
| `unordered_triples.txt` | Human-readable shuffled triples |
| `readable_graph.json` | Node-edge graph for programs |
| `readable_graph.md` | Human-readable adjacency list |

## Canonical relation direction

KQA Pro stores relations from both endpoint views and marks each view as
`forward` or `backward`. The script converts both views to one canonical
direction and deduplicates them. For example, these two stored views become
one statement:

```text
A --relation, forward--> B
B --relation, backward--> A

canonical: (A, relation, B)
```

## Alignment guarantee

The generator checks that all three representations contain exactly the same
set of statement IDs:

```text
canonical statement IDs
  = document statement IDs
  = triple statement IDs
  = graph edge IDs
```

If these sets differ, generation fails instead of silently producing an unfair
comparison.

## Scope

This is an entity-centred pilot subgraph, not yet a per-question gold evidence
extractor. It is suitable for validating serialization, indexing, prompt
formats, and graph readability. A later question-level stage should execute or
trace the gold KoPL program and render only the statements required by each
question.

## 涓枃锛?0棰?Oracle 瀹為獙浣跨敤璇存槑

`run_oracle_20.py` 浼氬浐瀹氶殢鏈虹瀛愶紝浠庤缁冮泦涓€夋嫨20閬撳彲鎵ц鐨勪袱璺抽棶棰樸€傞鐩?KoPL缁撴瀯涓猴細

```text
Find -> Relate -> FilterConcept -> Relate -> FilterConcept -> What
```

鑴氭湰杩借釜涓ゆ潯鏀寔绛旀鐨勫叧绯伙紝骞舵妸瀹屽叏鐩稿悓鐨勭煡璇嗗垎鍒〃绀烘垚鑷劧璇█鏂囨。銆佹棤搴?涓夊厓缁勫拰鍙Graph銆傚疄楠岄噰鐢ㄥ紑鏀惧紡鍥炵瓟锛氫笉鎶?0涓€欓€夐」浜ょ粰妯″瀷锛岃姹傛ā鍨?鐩存帴杩斿洖绛旀瀹炰綋鎴栨暟鍊硷紝鍐嶄緷鎹師濮嬪€欓€夐」鍜岄噾鏍囧噯绛旀杩涜瑙勮寖鍖栧尮閰嶃€傝繖鏍峰彲浠?鍑忓皯妯″瀷閫氳繃鎺掗櫎骞叉壈閫夐」鐚滅瓟妗堢殑褰卞搷銆?
### 绗竴姝ワ細鍙瀯寤哄拰妫€鏌?0棰橈紝涓嶈皟鐢ㄦā鍨?
```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_20.py `
  --build-only
```

缁撴灉浣嶄簬 `output/oracle_20/`銆傚叾涓細

- `oracle_questions.jsonl`锛?0閬撻銆侀噾鏍囧噯鍏崇郴鍜屼笁绉峅racle涓婁笅鏂囷紱
- `bundle_validation.json`锛氭暟鎹榻愭鏌ワ紱
- `metrics.json`锛氭瀯寤烘鏌ョ粨鏋滐紝姝ゆ椂 `llm_accuracy` 涓?`null`銆?
### 绗簩姝ワ細鍑嗗Qwen3 4B

鏈疄楠屼娇鐢∣llama涓殑 `qwen3:4b`锛?
```powershell
ollama pull qwen3:4b
ollama serve
```

濡傛灉Ollama宸茬粡鍦ㄥ悗鍙拌繍琛岋紝涓嶉渶瑕侀噸澶嶆墽琛?`ollama serve`銆傚彲浠ユ鏌ユā鍨嬶細

```powershell
ollama list
```

### 绗笁姝ワ細杩愯20棰樹笁绉嶈〃绀哄疄楠?
Ollama鎻愪緵OpenAI-compatible鎺ュ彛銆傛墽琛岋細

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:11434/v1"
$env:OPENAI_MODEL = "qwen3:4b"

.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_20.py
```

鎬诲叡浼氳皟鐢ㄦā鍨?0娆★細20棰樹箻浠?绉嶈〃绀恒€俀wen3鍙兘鍦ㄥ唴閮ㄤ娇鐢ㄤ竴閮ㄥ垎杈撳嚭棰勭畻杩涜
鎬濊€冿紝鍥犳鑴氭湰榛樿璁剧疆 `--max-tokens 512`銆傚闇€淇敼锛?
```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_20.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 512
```

### 杈撳嚭涓庢寚鏍?
- `predictions.jsonl`锛氭瘡娆℃ā鍨嬭皟鐢ㄧ殑绛旀銆佸師濮嬪搷搴斻€佹槸鍚︽纭拰token浣跨敤閲忥紱
- `metrics.json`锛氭渶缁堝熀纭€鎸囨爣銆?
鎸囨爣鍖呮嫭锛?
- `document`鍑嗙‘鐜囷紱
- `triple`鍑嗙‘鐜囷紱
- `graph`鍑嗙‘鐜囷紱
- 姣忕琛ㄧず鐨勬棤鏁堣緭鍑烘暟閲忥紱
- 鍚屼竴閬撻涓夌琛ㄧず鍏ㄩ儴鍥炵瓟姝ｇ‘鐨勬瘮渚嬶紱
- 浠诲姟妯″紡涓哄紑鏀惧紡绮剧‘鍖归厤锛屽洜姝や笉鎶ュ憡10閫?闅忔満鐚滄祴鍩虹嚎銆?
杩?0棰樺彧鏄祦绋嬪啋鐑熸祴璇曪紝涓嶉€傚悎鐩存帴浣滀负璁烘枃鏈€缁堢粨璁恒€傛寮忓疄楠屽簲鍥哄畾鏇村棰樼洰锛?骞舵姤鍛婄疆淇″尯闂村拰閰嶅鏄捐憲鎬ф楠屻€?
### 鏈満Qwen3 4B瀹炴祴缁撴灉

杩愯閰嶇疆锛?
```text
model: qwen3:4b (Ollama)
questions: 20
representations: document / triple / graph
calls: 60
temperature: 0
max_tokens: 512
task: 寮€鏀惧紡绛旀锛屼笌閲戞爣鍑嗙瓟妗堝仛瑙勮寖鍖栫簿纭尮閰?```

鍩虹缁撴灉锛?
| 琛ㄧず褰㈠紡 | 姝ｇ‘鏁?| 鍑嗙‘鐜?| 鏃犳晥杈撳嚭 |
| --- | ---: | ---: | ---: |
| Document | 17/20 | 85% | 3 |
| 鏃犲簭Triple | 16/20 | 80% | 4 |
| Graph | 16/20 | 80% | 4 |

涓夌褰㈠紡鍦ㄥ悓涓€閬撻涓婂叏閮ㄥ洖绛旀纭細15/20锛屽嵆75%銆?
闇€瑕佹敞鎰忥細杩欓噷鐨?1涓敊璇叏閮ㄨ璇勪及鍣ㄦ爣璁颁负鏃犳晥杈撳嚭銆傛鏌ュ師濮嬪搷搴斿悗鍙戠幇锛屽鏁版儏鍐?鏄疩wen3鍦?12 token棰勭畻鍐呬粛鍦ㄧ敓鎴愭€濊€冭繃绋嬶紝鏈€缁堢瓟妗堣鎴柇锛涙墍浠ユ湰缁撴灉鍚屾椂鍙楀埌
妯″瀷鏍煎紡閬靛惊鍜岃緭鍑洪绠楀奖鍝嶏紝涓嶈兘鐩存帴瑙ｉ噴鎴怐ocument涓€瀹氫紭浜嶵riple鎴朑raph銆?姝ｅ紡杩愯鍙互灏濊瘯 `--max-tokens 1024` 鎴栨洿楂橈紝骞跺悓鏃舵姤鍛婃棤鏁堣緭鍑虹巼銆?
## 涓枃锛?/2/3-hop鍒嗗眰Oracle瀹為獙锛堟殏鏈墽琛岋級

鏂板鑴氭湰 `run_oracle_hops_15.py`銆傚畠涓嶄細瑕嗙洊涓婃瀹為獙锛氭棫缁撴灉浠嶄繚瀛樺湪
`output/oracle_20/`锛屾柊缁撴灉榛樿鍐欏叆 `output/oracle_hops_15/`銆?
榛樿閫夋嫨鍗曚竴鍏崇郴閾剧殑1-hop銆?-hop銆?-hop闂鍚?閬擄紝鍏?5棰橈紱姣忛娴嬭瘯Document銆?鏃犲簭Triple鍜孏raph锛屽叡45娆℃ā鍨嬭皟鐢ㄣ€傚厑璁镐娇鐢?`And` 灏嗕腑闂村疄浣撲笌棰樼洰鏄庣‘缁欏嚭鐨?瀹炰綋瀵归綈锛屼絾鎺掗櫎涓ゆ潯鍏崇郴鍒嗘敮骞惰姹囧悎銆侀泦鍚堝拰姣旇緝绋嬪簭銆傝烦鏁版寜渚濊禆DAG涓渶闀跨殑
`Relate` 閾捐绠楋紝鑰屼笉鏄畝鍗曠粺璁＄▼搴忔楠ゆ暟銆?
鍙瀯寤哄拰妫€鏌ユ娊鏍凤紝涓嶈皟鐢ㄦā鍨嬶細

```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_hops_15.py `
  --build-only
```

浣跨敤鏈満Ollama鐨?`qwen3:4b` 鎵ц锛?
```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 1024
```

瀹屾暣鍙傛暟褰㈠紡锛?
```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_hops_15.py `
  --hops 1 2 3 `
  --per-hop 5 `
  --sample-seed 42 `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 1024
```

鏂?`metrics.json` 浼氳褰曟€讳綋琛ㄧず鍑嗙‘鐜囷紝浠ュ強姣忎釜hop涓嬬殑Document銆乀riple銆丟raph
鍑嗙‘鐜囥€佹棤鏁堣緭鍑虹巼銆佹帓闄ゆ棤鏁堣緭鍑哄悗鐨勫噯纭巼鍜屼笁绉嶈〃绀哄叏閮ㄧ瓟瀵圭巼銆?
