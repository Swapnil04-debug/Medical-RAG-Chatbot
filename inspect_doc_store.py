import pickle

with open("data/doc_store.pkl", "rb") as f:
    doc_store = pickle.load(f)

print("Type of doc_store:", type(doc_store))

if isinstance(doc_store, dict):
    keys = list(doc_store.keys())[:5]
    print("Sample keys:", keys, "key type:", type(keys[0]) if keys else None)
    sample_val = doc_store[keys[0]]
    print("Sample value type:", type(sample_val))
    print("Sample value:", sample_val)
elif isinstance(doc_store, list):
    print("Length:", len(doc_store))
    print("Sample item type:", type(doc_store[0]))
    print("Sample item:", doc_store[0])
else:
    print("Repr:", repr(doc_store)[:500])