import json
import pickle

with open("data/doc_store.pkl", "rb") as f:
    doc_store = pickle.load(f)

max_idx = max(doc_store.keys())
converted = [None] * (max_idx + 1)

for idx, doc in doc_store.items():
    converted[idx] = {
        "page_content": doc.page_content,
        "metadata": dict(doc.metadata or {}),
    }

missing = [i for i, v in enumerate(converted) if v is None]
if missing:
    print(f"Warning: {len(missing)} indices missing, filling with empty docs: {missing[:10]}...")
    for i in missing:
        converted[i] = {"page_content": "", "metadata": {}}

with open("data/doc_store.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False)

print(f"Converted {len(converted)} documents to data/doc_store.json")