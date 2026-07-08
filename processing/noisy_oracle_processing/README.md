# Gold + Noise 鍚櫔澹板疄楠?
杩欐槸鐙珛鐨勭涓夌粍瀹為獙锛屼笉淇敼浠ヤ笅鐜版湁瀹為獙锛?
```text
knowledge_representation_processing/   # Gold-only Oracle
rag_full_kb_processing/                # 瀹屾暣KB鐪熷疄RAG
```

鏈疄楠岃鍙栧悓涓€鎵?/2/3-hop鍚?棰橈紝淇濊瘉鎵€鏈塆old浜嬪疄瀛樺湪锛屽啀涓烘瘡鏉old浜嬪疄鍔犲叆5鏉?鍣０銆傚櫔澹颁紭鍏堥€夊彇鍏变韩瀹炰綋鎴栫浉鍚屽叧绯荤殑鍥伴毦浜嬪疄锛屼笉瓒虫椂浠庡畬鏁碖B鐨勫浐瀹氶殢鏈烘睜琛ュ厖銆?鍚屼竴閬撻鐨凞ocument銆乀riple鍜孏raph鍏变韩瀹屽叏鐩稿悓鐨凣old/Noise statement IDs銆?
## 鍙敓鎴愬拰妫€鏌ュ惈鍣暟鎹?
```powershell
.\venv\Scripts\python.exe `
  processing\noisy_oracle_processing\run_gold_noise_hops_15.py `
  --build-only
```

## 浣跨敤qwen3:4b杩愯

```powershell
.\venv\Scripts\python.exe `
  processing\noisy_oracle_processing\run_gold_noise_hops_15.py `
  --noise-per-gold 5 `
  --noise-seed 42 `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 1024
```

榛樿杈撳嚭鍒帮細

```text
noisy_oracle_processing/output/noise_1_to_5/
```

鍏朵腑锛?
- `noise_questions.jsonl`锛氭瘡棰楪old IDs銆丯oise IDs銆佷笁绉嶅惈鍣笂涓嬫枃锛?- `bundle_validation.json`锛氭暟鎹榻愪笌闅旂妫€鏌ワ紱
- `predictions.jsonl`锛氭ā鍨嬮€愰杈撳嚭锛?- `metrics.json`锛氭寜琛ㄧず褰㈠紡鍜宧op缁熻鍑嗙‘鐜囧強鏃犳晥杈撳嚭鐜囥€?
姝ｅ紡鎵╁ぇ瀹為獙鏃讹紝鍙互鍒嗗埆杩愯 `--noise-per-gold 1`銆乣5`銆乣10`锛屼絾蹇呴』涓轰笉鍚?鍣０姣斾緥璁剧疆涓嶅悓杈撳嚭鐩綍锛屽苟淇濇寔棰樼洰銆侀殢鏈虹瀛愬拰妯″瀷鍙傛暟涓€鑷淬€?
