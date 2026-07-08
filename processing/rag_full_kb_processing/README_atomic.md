# E3 Atomic RAG锛堜弗鏍煎榻怌hunk锛?
Atomic鐗堟湰鐢ㄤ簬鎺у埗chunk绮掑害瀵规绱㈢粨鏋滅殑褰卞搷銆侱ocument銆乀riple鍜孏raph閮界敱鍚屼竴浠絚anonical statements鐢熸垚锛屽苟涓ユ牸婊¤冻锛?
```text
涓€涓猻tatement ID = 涓€涓猚hunk
```

鍚屼竴浜嬪疄鐨勪笁涓猚hunk鍏变韩鐩稿悓`chunk_id`鍜宍statement_id`锛屽彧鏀瑰彉鏂囨湰琛ㄧず锛?
```text
Document: Paris has the relation 'capital of' to France.
Triple:   (Paris | capital of | France)
Graph:    EDGE Q1 (Paris) --[capital of]--> Q2 (France)
```

妫€绱㈡椂涓夌琛ㄧず浣跨敤鐩稿悓BM25瀹炵幇锛屽苟鍒嗗埆鍙朤op-N鏉tatement銆傞粯璁Top 100`锛屼笉闄愬埗鐩稿悓token锛沺rompt token鍜屽欢杩熶綔涓哄疄楠岀粨鏋滆褰曘€?
## 1. 鏋勫缓Atomic璇枡

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\rag_full_kb_processing\build_atomic_kb_corpora.py
```

杈撳嚭浣嶄簬`output/atomic_kb/`銆傝姝ラ涓嶈鍙栭鐩拰Gold锛屼篃涓嶈皟鐢ㄦā鍨嬨€?
## 2. 鍙繍琛屾绱?
```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\rag_full_kb_processing\run_rag_atomic_hops_15.py `
  --retrieval-only `
  --top-statements 100
```

## 3. 浣跨敤16K妯″瀷杩愯瀹屾暣E3

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\rag_full_kb_processing\run_rag_atomic_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096 `
  --top-statements 100
```

榛樿缁撴灉淇濆瓨鍒癭output/rag_atomic_hops_15/`锛屼笉浼氳鐩栧師鏉ョ殑Native RAG缁撴灉銆?
## 鎸囨爣

- `statement_recall`锛氳繘鍏ユ渶缁圱op-N涓婁笅鏂囩殑Gold statement姣斾緥銆?- `complete_path_recall`锛氬畬鏁碐old璺緞鏄惁杩涘叆涓婁笅鏂囥€?- `retrieval_mrr`锛欸old鍦ㄦ渶澶?000鏉℃绱㈡帓搴忎腑鐨凪RR銆?- `context_mrr`锛欸old鍦ㄦ渶缁圱op-N涓婁笅鏂囦腑鐨凪RR銆?- `mean_context_statements`锛氬簲涓?00锛岄櫎闈炴绱㈢粨鏋滀笉瓒炽€?- 妯″瀷鍑嗙‘鐜囥€佹棤鏁堣緭鍑恒€乸rompt/completion tokens鍜屽欢杩熴€?
褰撳墠Native鐗堟湰缁х画淇濈暀锛岀敤浜庢祴閲忕湡瀹炲垎鍧楃郴缁熸晥鏋滐紱Atomic鐗堟湰浣滀负闅旂鐭ヨ瘑琛ㄧず褰卞搷鐨勪富瑕丒3鏉′欢銆?
