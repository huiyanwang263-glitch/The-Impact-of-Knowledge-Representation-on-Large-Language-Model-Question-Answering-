# 瀹屾暣KB鐨凴AG瀹為獙璇枡

鏈洰褰曞彧璐熻矗鎶婂畬鏁?`kb.json` 杞崲鎴愪笁绉嶅彲寤虹珛绱㈠紩鐨勭煡璇嗚〃绀猴紝涓嶈繍琛屾绱㈠櫒锛?涓嶈皟鐢↙LM锛屼篃涓嶈鍙栭棶棰樼瓟妗堟垨Gold瀛愬浘銆?
```text
瀹屾暣KB
  -> canonical statements
     -> Document chunks
     -> 鍏ㄥ眬鏃犲簭Triple chunks
     -> Graph nodes / edges / adjacency chunks
```

涓夊璇枡鍏变韩瀹屽叏鐩稿悓鐨?`statement_id`銆傛湭鏉ラ€夊畾1000棰樺悗锛屽叾Gold瀛愬浘鍙敤浜庤绠?`Recall@k`銆佸畬鏁磋矾寰勮鐩栫巼绛夋寚鏍囷紝涓嶈繘鍏ョ储寮曪紝涔熶笉鐢ㄤ簬绛涢€夊€欓€夌煡璇嗐€?
## 鐢熸垚鍛戒护

鍦ㄥ伐浣滃尯鏍圭洰褰曡繍琛岋細

```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\build_full_kb_corpora.py
```

榛樿杈撳嚭鍒帮細

```text
processing/rag_full_kb_processing/output/full_kb/
```

## 杈撳嚭鏂囦欢

| 鏂囦欢 | 鐢ㄩ€?|
| --- | --- |
| `manifest.json` | 瑙勬ā銆佸弬鏁板強涓夌琛ㄧず瀵归綈鏍￠獙 |
| `canonical/statements.jsonl` | 鍞竴瑙勮寖浜嬪疄灞傦紝涓嶇洿鎺ヤ綔涓哄疄楠屾潯浠?|
| `document/documents.jsonl` | 鑷劧璇█鐭ヨ瘑鍧楋紝鍙洿鎺ュ仛鍚戦噺/BM25绱㈠紩 |
| `triple/unordered_triples.jsonl` | 鍥哄畾闅忔満绉嶅瓙鎵撲贡鐨勫崟浜嬪疄涓夊厓缁勫潡 |
| `graph/nodes.jsonl` | Graph鑺傜偣琛?|
| `graph/edges.jsonl` | Graph杈硅〃锛屾瘡鏉¤竟瀵瑰簲涓€涓猻tatement ID |
| `graph/adjacency_chunks.jsonl` | 鍙銆佸彲宓屽叆鐨勯偦鎺ヨ〃鏂囨湰鍧?|

Document鍜孏raph閭绘帴琛ㄩ粯璁ゆ瘡20鏉′簨瀹炲垎鍧楋紝閬垮厤楂樿繛鎺ュ疄浣撲骇鐢熻秴闀夸笂涓嬫枃銆傚彲浠ョ敤
`--chunk-size` 淇敼锛屼絾涓夌瀹為獙蹇呴』璁板綍鍚勮嚜token鏁伴噺锛屽苟浣跨敤鐩稿悓鐨勪笂涓嬫枃token
棰勭畻锛岃€屼笉鏄畝鍗曞浐瀹氱浉鍚?`top-k`銆?
```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\build_full_kb_corpora.py `
  --chunk-size 20 `
  --shuffle-seed 42
```

## 鍚庣画绱㈠紩鍘熷垯

- Document鏉′欢绱㈠紩 `document/documents.jsonl` 鐨?`text`锛?- Triple鏉′欢绱㈠紩 `triple/unordered_triples.jsonl` 鐨?`text`锛?- Graph鏉′欢淇濈暀 `nodes/edges` 鍋氬浘閬嶅巻锛屽悓鏃跺彲绱㈠紩 `adjacency_chunks.jsonl`锛?- 涓夌鏉′欢蹇呴』浣跨敤鍚屼竴闂闆嗗悎銆佸悓涓€妯″瀷鍜屽悓涓€涓婁笅鏂噒oken棰勭畻锛?- Gold statement IDs鍙兘鐢ㄤ簬绂荤嚎璇勪环锛屼笉寰楀弬涓庡彫鍥炪€侀噸鎺掓垨涓婁笅鏂囨瀯閫犮€?
## 1/2/3-hop鍚岄RAG娴嬭瘯鑴氭湰锛堟殏鏈墽琛岋級

`run_rag_hops_15.py` 鐩存帴璇诲彇Oracle瀹為獙宸茬粡鍥哄畾鐨勫悓涓€鎵?5棰橈紝涓嶉噸鏂版娊棰橈細

```text
knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl
```

Document銆乀riple鍜孏raph缁熶竴浣跨敤鑴氭湰鍐呭悓涓€濂桞M25瀹炵幇锛屽寘鎷浉鍚屽垎璇嶃€佸弬鏁般€佸師濮?闂鏌ヨ銆佺ǔ瀹氭帓搴忚鍒欏拰2000璇嶄笂涓嬫枃棰勭畻銆侴old statement IDs鍦ㄦ绱㈠拰涓婁笅鏂囬€夋嫨
瀹屾垚涔嬪悗鎵嶇敤浜庤绠桼ecall锛屼笉鍙備笌绱㈠紩銆佹墦鍒嗐€侀噸鎺掓垨Prompt鏋勯€犮€?
鍙繍琛屾绱㈣瘎浼帮紝涓嶈皟鐢↙LM锛?
```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\run_rag_hops_15.py `
  --retrieval-only
```

瀹屾暣RAG娴嬭瘯锛屼娇鐢ㄦ湰鏈篛llama鐨?`qwen3:4b`锛?
```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\run_rag_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 1024 `
  --context-word-budget 2000
```

榛樿缁撴灉鍐欏叆 `output/rag_hops_15/`锛?
- `retrieval_results.jsonl`锛氭绱㈠潡銆佸垎鏁般€乻tatement IDs鍜岄€愰Recall锛?- `predictions.jsonl`锛氳繍琛孡LM鏃剁殑绛旀锛?- `metrics.json`锛氭寜hop鍜岃〃绀哄舰寮忔眹鎬绘绱笌绛旀鎸囨爣锛?- `configuration.json`锛氱粺涓€妫€绱㈠弬鏁般€?
绗竴鐗堜娇鐢˙M25鏄负浜嗕弗鏍间繚璇佷笁绉嶈〃绀洪噰鐢ㄧ浉鍚屾绱㈡祦绋嬨€備互鍚庤嫢鎹㈡垚Embedding锛?搴斿涓夌琛ㄧず浣跨敤鍚屼竴涓狤mbedding妯″瀷銆佺浉鍚岀浉浼煎害鍙婄浉鍚宼oken棰勭畻锛屽苟瑙嗕负鍙︿竴缁?鐙珛瀹為獙閰嶇疆銆?
