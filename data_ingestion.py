import re
import polars as pl
from typing import List, Dict
from qdrant_client import QdrantClient, models

df = pl.read_csv("data/japantravel_posts_with_comments.csv")

df = df.with_columns(pl.col('selftext').fill_null(''),
                     pl.col('comment1').fill_null(''),
                     pl.col('comment2').fill_null(''),
                     pl.col('comment3').fill_null(''),
                     pl.col('comment4').fill_null(''),
                     pl.col('comment5').fill_null(''))
df = df.with_columns(
    (
        "Comment 1: " + pl.col("comment1") + "\n" +
        "Comment 2: " + pl.col("comment2") + "\n" +
        "Comment 3: " + pl.col("comment3") + "\n" +
        "Comment 4: " + pl.col("comment4") + "\n" +
        "Comment 5: " + pl.col("comment5")
    ).alias("comments_combined")
    ).drop(["comment1", "comment2", "comment3", "comment4", "comment5"])

### Sample dataset due to limits for llm
RANDOM_SEED = 42
n_samples = 80
df_sampled = df.sample(n=n_samples, seed=RANDOM_SEED)
documents = df_sampled.to_dicts()
# documents = df.to_dicts()

### Initiate client
# qd_client = QdrantClient("http://localhost:6333")
qd_client = QdrantClient("http://host.docker.internal")
# qd_client = QdrantClient("http://qdrant:6333")

### Create the collection with specified sparse vector parameters
collection_name = "travel-rec-dense-and-sparse"
qd_client.delete_collection(collection_name)

EMBEDDING_DIMENSIONALITY = 512
model_handle = "jinaai/jina-embeddings-v2-small-en"


qd_client.create_collection(
    collection_name=collection_name,
    vectors_config = {
        "jina-small": models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
        ),        
    },
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        ),
    }
)

def chunk_text(
    text: str, 
    doc_id: str, 
    chunk_size: int = 200, 
    overlap: int = 20
) -> List[Dict]:
    """Split text into overlapping chunks of words with unique IDs."""
    words = re.split(r"\s+", text.strip())
    
    chunks = []
    start = 0
    chunk_num = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "id": f"{doc_id}_{chunk_num}",
            'doc_id': doc_id,
            "text": chunk_text
        })

        chunk_num += 1

        if end == len(words) or len(words) - end <= overlap:
            break
        start = end - overlap
    
    return chunks

def upsert_documents_hybrid(collection_name, docs):

    all_chunks = []
    for doc in docs:
        text = f"Title: {doc['title']}\n\nContent: {doc['selftext']}\n\nComments:\n{doc['comments_combined']}"
        chunks = chunk_text(text, doc['id'], chunk_size=200, overlap=20)
        all_chunks.extend(chunks)

        # Build Qdrant points
    points = [
        models.PointStruct(
            id=i, 
            vector={
                "jina-small": models.Document(text = doc['text'], model = "jinaai/jina-embeddings-v2-small-en"),
                "bm25":models.Document(text = doc['text'], model = "Qdrant/bm25")},
            payload={"id": doc["id"], "doc_id": doc["doc_id"], "text": doc["text"]}
            
        )
        for i, doc in enumerate(all_chunks)
    ]

        # Upsert into Qdrant
    qd_client.upsert(
        collection_name=collection_name,
        points=points
    )





upsert_documents_hybrid(collection_name, documents)