import pickle
from collections import defaultdict

# -----------------------------
# Load document store
# -----------------------------
with open("data/doc_store.pkl", "rb") as f:
    doc_store = pickle.load(f)

print(f"Loaded {len(doc_store)} documents.\n")


def matches_query_terms(text):
    t = text.lower()
    return any(term in t for term in ["migraine", "headache"])


# -----------------------------
# Breakdown by source/type
# -----------------------------
candidates = []
for idx, doc in doc_store.items():
    if matches_query_terms(doc.page_content):
        candidates.append((idx, doc))

by_source = defaultdict(list)
for idx, doc in candidates:
    src = doc.metadata.get("source", "Unknown")
    typ = doc.metadata.get("type", "unknown")
    by_source[(src, typ)].append((idx, doc))

print("========== Breakdown by Source ==========\n")
for (src, typ), items in sorted(by_source.items(), key=lambda x: -len(x[1])):
    print(f"{src} [{typ}] : {len(items)} chunks")


# -----------------------------
# FAISS retrieval simulation
# -----------------------------
print("\n========== FAISS Retrieval ==========\n")

import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
index = faiss.read_index("data/faiss_index.bin")

queries = ["what is migraine", "what is Migraine"]

for query in queries:
    print("=" * 80)
    print("Query:", query)
    print("=" * 80)

    emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(emb.astype("float32"), k=10)

    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        doc = doc_store[idx]
        print(f"\nRank {rank}")
        print(f"Distance : {score:.4f}")
        print(f"Source   : {doc.metadata.get('source')}")
        print(f"Type     : {doc.metadata.get('type')}")
        print(doc.page_content[:250].replace("\n", " "))