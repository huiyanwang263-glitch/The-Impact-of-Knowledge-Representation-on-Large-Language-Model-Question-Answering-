# E0 Closed-book瀹為獙

鏈洰褰曟槸瀹屽叏鐙珛鐨凟0鍩虹嚎锛屼笉淇敼宸叉湁瀹為獙锛?
```text
knowledge_representation_processing/  # E1 Gold-only
noisy_oracle_processing/              # Controlled Gold + Noise
rag_full_kb_processing/               # E3 Real RAG
```

E0璇诲彇鍚屼竴鎵?/2/3-hop鍚?棰橈紝浣哖rompt涓彧鎻愪緵闂锛屼笉鎻愪緵Gold浜嬪疄銆佸畬鏁碖B銆佹绱?缁撴灉鎴栧€欓€夌瓟妗堛€傚畠鐢ㄤ簬浼拌qwen3:4b渚濊禆鍙傛暟鐭ヨ瘑銆侀鍨嬫ā寮忔垨鐚滄祴鑳藉绛斿澶氬皯棰樸€?
## 鍙瀯寤哄苟妫€鏌ラ鐩?
```powershell
.\venv\Scripts\python.exe `
  processing\closed_book_processing\run_e0_closed_book_hops_15.py `
  --build-only
```

## 浣跨敤qwen3:4b杩愯E0

```powershell
.\venv\Scripts\python.exe `
  processing\closed_book_processing\run_e0_closed_book_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --max-tokens 4096
```

榛樿杈撳嚭鐩綍锛?
```text
closed_book_processing/output/e0_hops_15/
```

杈撳嚭鍖呮嫭锛?
- `closed_book_questions.jsonl`锛氫粎淇濈暀闂涓庤瘎浠峰瓧娈碉紱
- `bundle_validation.json`锛氱‘璁ゆ棤鐭ヨ瘑涓婁笅鏂囥€佹棤妫€绱€佹棤鍊欓€夐」杈撳叆锛?- `predictions.jsonl`锛氶€愰杈撳嚭銆乼oken鐢ㄩ噺鍜屾ā鍨嬪欢杩燂紱
- `metrics.json`锛氭€讳綋鍙婃寜1/2/3-hop缁熻鐨勫噯纭巼銆乼oken鍜屽欢杩熴€?
E0娌℃湁Document銆乀riple銆丟raph涓変釜鏉′欢锛屽洜涓烘病鏈夊閮ㄧ煡璇嗚〃绀恒€傚畠鍙綔涓轰笁绉嶈〃绀?瀹為獙鍏变韩鐨勫悓涓€涓弬鏁扮煡璇嗗熀绾裤€?
## 鏈満qwen3:4b瀹炴祴缁撴灉

閰嶇疆锛?5棰樸€佹棤涓婁笅鏂囥€佹棤鍊欓€夐」銆乣temperature=0`銆乣max_tokens=4096`銆?
| Hop | 姝ｇ‘鏁?| 鍑嗙‘鐜?| 鏈尮閰嶅埌鍘熷绛旀閫夐」 |
| --- | ---: | ---: | ---: |
| 1-hop | 2/5 | 40% | 3/5 |
| 2-hop | 1/5 | 20% | 3/5 |
| 3-hop | 1/5 | 20% | 4/5 |
| 鎬讳綋 | 4/15 | 26.67% | 10/15 |

鎬讳綋骞冲潎姣忛浣跨敤64.53涓緭鍏oken銆?373.07涓緭鍑簍oken锛屾€昏1437.60涓猼oken锛?骞冲潎妯″瀷寤惰繜绾?7.32绉掋€傝繖閲岀殑鈥滄湭鍖归厤鈥濊〃绀烘ā鍨嬪紑鏀惧紡绛旀鏈尮閰嶄换浣曞師濮嬪€欓€夐」锛?涓嶅簲鍏ㄩ儴瑙ｉ噴鎴愭牸寮忛敊璇紝瀹冧篃鍖呭惈妯″瀷鍦ㄦ棤鐭ヨ瘑鏉′欢涓嬬粰鍑洪敊璇疄浣撶殑鎯呭喌銆?
