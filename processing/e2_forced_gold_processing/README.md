# E2锛氱湡瀹炴绱㈠櫔澹颁笅鐨勫己鍒?Gold 瀹為獙

鏈洰褰曟槸鐙珛鐨?E2 瀹為獙锛屼笉淇敼宸叉湁鐨?E0銆丟old-only銆佸彈鎺у櫔澹板拰鐪熷疄 RAG 瀹為獙銆?
## 瀹為獙鐩殑

E2 鐢ㄤ簬鐮旂┒锛氬綋妫€绱㈠凡缁忔垚鍔熴€佸畬鏁?Gold 鎺ㄧ悊璺緞涓€瀹氬瓨鍦ㄦ椂锛屼笉鍚岀煡璇嗚〃绀烘槸鍚﹀奖鍝?LLM 浠庡惈鍣笂涓嬫枃涓瘑鍒苟浣跨敤璇佹嵁銆?
瀹冧綅浜?E1 鍜?E3 涔嬮棿锛?
- E1 Gold-only锛氬彧鏈?Gold锛屼富瑕佹祴琛ㄧず褰㈠紡瀵规帹鐞嗙殑褰卞搷銆?- E2 Forced-Gold锛欸old 瀹屾暣瀛樺湪锛屽悓鏃朵繚鐣欑湡瀹?RAG 鍣０锛屾祴璇佹嵁閫夋嫨涓庝娇鐢ㄣ€?- E3 Real RAG锛氫笉琛?Gold锛屾祴妫€绱㈠拰璇佹嵁浣跨敤鐨勭患鍚堟晥鏋溿€?
## E2 涓婁笅鏂囧浣曟瀯閫?
鑴氭湰璇诲彇 E3 宸蹭繚瀛樼殑 `retrieval_results.jsonl`锛屼笉浼氶噸鏂拌繍琛?BM25锛?
1. 瀵规瘡閬撻鍜屾瘡绉嶈〃绀猴紝鎸夋绱㈡帓鍚嶅睍寮€ statement IDs锛屽苟鍘婚櫎閲嶅椤广€?2. 妫€鏌ュ摢浜?Gold statement IDs 宸茬粡琚湡瀹?RAG 鍙洖銆?3. 淇濈暀宸插彫鍥炵殑 Gold 鍜屽師鏈夊櫔澹般€?4. 姣忕己灏戜竴鏉?Gold锛屽氨鍒犻櫎涓€鏉℃渶浣庢帓鍚嶇殑闈?Gold 璇彞锛屽苟鍦ㄥ悓涓€浣嶇疆鎻掑叆璇?Gold銆?5. 鏇挎崲鍓嶅悗 statement 鏁伴噺涓ユ牸鐩稿悓锛屾渶缁?Gold 璺緞鍙洖鐜囦负 100%銆?6. 鎸?Document銆乁nordered Triple銆丟raph 鍒嗗埆閲嶆柊娓叉煋涓婁笅鏂囥€侴raph浣跨敤绱у噾鐨勫唴鑱旀爣绛捐竟鏍煎紡锛?
```text
EDGE Q1 (瀹炰綋A) --[鍏崇郴]--> Q2 (瀹炰綋B)
```

Graph涓嶅啀鍏堣緭鍑虹嫭绔嬭妭鐐硅〃锛岄伩鍏嶈妭鐐硅〃鍗犳弧涓婁笅鏂囧悗鎶婂叧閿竟鎴柇銆?
璇ユ牸寮忕幇宸蹭綔涓轰笁涓惈鐭ヨ瘑闃舵鐨勭粺涓€Graph鏍煎紡锛欵1 Gold-only銆丒2 Forced-Gold鍜孍3 Real RAG銆傚疄浣揑D鐢ㄤ簬鏄庣‘鑺傜偣澶嶇敤锛屽疄浣撳悕绉扮敤浜庤LLM璇诲彇璇箟锛涢檺瀹氱缁х画闄勫湪瀵瑰簲杈瑰悗銆?
榛樿閲囩敤 `native noise`锛氫笁绉嶈〃绀哄垎鍒繚鐣欒嚜宸卞湪 E3 涓疄闄呮绱㈠埌鐨勫櫔澹般€傝繖浣?E2 鑳戒笌 E3 鍦ㄧ浉鍚屸€滆〃绀?+ 鐪熷疄妫€绱㈠櫔澹扳€濇潯浠朵笅杩涜閫愰姣旇緝銆備笁绉嶈〃绀洪棿鐨勫櫔澹板苟涓嶅畬鍏ㄧ浉鍚岋紝鍥犳 E2 閫傚悎瀹氫綅绔埌绔崯澶憋紝涓嶅簲鍗曠嫭褰撲綔鏈€涓ユ牸鐨勭函琛ㄧず鍥犳灉瀹為獙锛涗弗鏍艰〃绀哄鐓т粛鐢?E1 鎴栫幇鏈夊彈鎺у櫔澹板疄楠屾壙鎷呫€?
## 杈撳叆鍜岃緭鍑?
榛樿杈撳叆锛?
- 15閬撳浐瀹氶锛歚../knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl`
- E3妫€绱㈢粨鏋滐細`../rag_full_kb_processing/output/rag_hops_15/retrieval_results.jsonl`
- 瀹屾暣KB瑙勮寖璇彞锛歚../rag_full_kb_processing/output/full_kb/canonical/statements.jsonl`

榛樿杈撳嚭鍒?`output/e2_hops_15/`锛?
- `e2_contexts.jsonl`锛?5涓瀯閫犲悗鐨勪笂涓嬫枃鍙婃浛鎹㈣褰曘€?- `bundle_validation.json`锛欸old瀹屾暣鎬с€佽鍙ユ暟瀹堟亽绛夋鏌ャ€?- `predictions.jsonl`锛氬疄闄呰皟鐢ㄦā鍨嬪悗鐢熸垚銆?- `metrics.json`锛氭寜琛ㄧず鍜岃烦鏁版眹鎬诲噯纭巼銆乼oken涓庡欢杩熴€?
姣忔潯璁板綍淇濈暀鍘熷鍙洖鎸囨爣銆佺己澶盙old銆佽鍒犻櫎鐨勫櫔澹般€佹浛鎹綅缃€佹渶缁坰tatement IDs鍜屽畬鏁翠笂涓嬫枃锛屼究浜庡璁°€?
鍙﹀淇濆瓨 `source_replay_context`锛氬畠浣跨敤鏇挎崲鍓嶇殑鍘熷妫€绱?statement IDs锛岀粡杩囦笌E2瀹屽叏鐩稿悓鐨勮鑼冨寲娓叉煋銆傛棫E3鐩存帴浣跨敤璇枡搴撲腑鐨勫潡鏂囨湰锛屽叾涓璂ocument鍚疄浣撴爣棰樸€丟raph鍙兘閲嶅鑺傜偣鍖猴紱E2鍒欓渶瑕佸湪璇彞绾ф浛鎹㈠悗閲嶆柊娓叉煋銆傚洜姝わ紝鑻ユ寮忓垎鏋怑2涓嶦3涔嬮棿鐨勫噯纭巼宸€硷紝搴斿鍔犱竴娆′娇鐢?`source_replay_context` 鐨凟3閲嶆斁锛屼娇浜岃€呭彧鐩稿樊Gold琛ラ綈銆備笉瑕佺洿鎺ユ妸鏃3鍑嗙‘鐜囦笌E2宸€煎叏閮ㄨВ閲婁负Gold缂哄け閫犳垚銆?
## 浠呮瀯閫犲苟妫€鏌ヤ笂涓嬫枃

浠ヤ笅鍛戒护涓嶄細璋冪敤妯″瀷锛?
```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\e2_forced_gold_processing\run_e2_forced_gold_hops_15.py `
  --build-only
```

## 姝ｅ紡杩愯锛堝綋鍓嶄笉瑕佹墽琛岋級

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\e2_forced_gold_processing\run_e2_forced_gold_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 4096
```

鑴氭湰鍏辫皟鐢ㄦā鍨?`15棰?脳 3绉嶈〃绀?= 45娆銆傛病鏈夋彁渚?`--base-url` 鍜?`--model` 鏃朵篃鍙瀯閫犳暟鎹紝涓嶄細璋冪敤妯″瀷銆?
### 鍙繍琛岀揣鍑慓raph锛?5娆¤皟鐢級

浣跨敤鐙珛杈撳嚭鐩綍锛岄伩鍏嶈鐩栧師鏉ョ殑涓夎〃绀虹粨鏋滐細

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\e2_forced_gold_processing\run_e2_forced_gold_hops_15.py `
  --representations graph `
  --output-dir processing\e2_forced_gold_processing\output\e2_graph_inline_hops_15 `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 4096
```

### 浣跨敤16K涓婁笅鏂囬噸鏂拌繍琛岀揣鍑慓raph

`--max-tokens`鍙帶鍒舵ā鍨嬫渶澶ц緭鍑洪暱搴︼紝涓嶈兘鎵╁ぇOllama鐨勮緭鍏ヤ笂涓嬫枃銆侽penAI鍏煎鎺ュ彛闇€瑕佸厛鐢∕odelfile鍒涘缓涓€涓?6K涓婁笅鏂囨ā鍨嬶細

```powershell
ollama create qwen3-4b-16k `
  -f processing\e2_forced_gold_processing\Modelfile.qwen3-4b-16k
```

鐒跺悗浣跨敤鏂扮殑妯″瀷鍚嶈繍琛岋紝骞跺啓鍏ユ柊鐨勭洰褰曚互淇濈暀涔嬪墠缁撴灉锛?
```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\e2_forced_gold_processing\run_e2_forced_gold_hops_15.py `
  --representations graph `
  --output-dir processing\e2_forced_gold_processing\output\e2_graph_inline_16k_hops_15 `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096
```

杩愯寮€濮嬪悗鍙互鍦ㄥ彟涓€涓粓绔鏌ュ疄闄呭垎閰嶇殑涓婁笅鏂囷細

```powershell
ollama ps
```

鏂扮殑Graph缁撴灉涓紝`mean_prompt_tokens`涓嶅簲鍐嶅浐瀹氫负2050銆?6K浼氬鍔犳樉瀛樻垨鍐呭瓨鍗犵敤锛涘鏋滃彂鐢熸樉瀛樹笉瓒虫垨澶ч噺CPU鍗歌浇锛屽彲鍏堟敼涓?192銆?
## 涓昏鎺у埗鍙橀噺

- 涓嶦0/E1/E3浣跨敤瀹屽叏鐩稿悓鐨?5閬撻鍜?/2/3-hop鍒掑垎銆?- E2鐩存帴浣跨敤E3宸茬粡浜х敓鐨勬绱㈢粨鏋滐紝涓嶆敼鍙樻绱㈠櫒銆?- Gold鍙湪妫€绱㈢粨鏉熷悗琛ュ叆锛屼笉鍙備笌BM25鎺掑簭銆?- 鍚屼竴鏉2璁板綍鏇挎崲鍓嶅悗璇彞鏁伴噺鐩稿悓銆?- 涓嶅己鍒朵笁绉嶈〃绀哄叿鏈夌浉鍚宼oken鏁帮紱token鏁板拰寤惰繜浣滀负瀹為獙缁撴灉璁板綍銆?- 鎻愮ず璇嶇殑鍥炵瓟瑕佹眰涓嶦3淇濇寔涓€鑷达紝涓嶅悜妯″瀷鎻愪緵閫夐」銆?- 姝ｅ紡鐨凟2/E3閰嶅姣旇緝搴斾娇鐢ㄥ悓涓€瑙勮寖鍖栨覆鏌擄紱`source_replay_context`宸蹭负姝ら鐣欍€?
