from qdrant_client import QdrantClient, models
from data_ingestion import qd_client, collection_name

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv() 
client = OpenAI()

# search
def hybrid_search(query, limit):
    results = qd_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=models.Document(
                text=query,
                model='jinaai/jina-embeddings-v2-small-en',
            ),
            using='jina-small',
            limit=limit,
            ),
            models.Prefetch(query=models.Document(
                text=query,
                model='Qdrant/bm25',
            ),
            using='bm25',
            limit=limit)
        ],
        
        query = models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True
    )
    return results.points

def build_prompt_vector_search(search_results, query):
    prompt_template = """"
    You are a helpful travel guide for people visiting Japan. 
    Answer the QUESTION based on the CONTEXT from travel discussions and experiences as concise as possible.  
    Use only the facts from the CONTEXT when you are answering the QUESTION.  
    If the CONTEXT does not contain enough information, say that you don’t know.  

    QUESTION: {question}  

    CONTEXT: {context}

    """.strip()

    max_docs = len(search_results)
    context_parts = []
    for i, result in enumerate(search_results[:max_docs], start=1):
        doc = result.payload
        part = f"""
            Document {i}:
            Text: {doc.get('text', '')}
            Doc_id: {doc.get('doc_id', '')}
            Id: {doc.get('id', '')}  
        """
        context_parts.append(part.strip())
        
    context = "\n\n".join(context_parts)
    prompt = prompt_template.format(question=query, context=context)
    return prompt


def llm(prompt):
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag_hybrid_search(query):
    search_results = hybrid_search(query, 1)
    prompt = build_prompt_vector_search(search_results, query) # check
    answer = llm(prompt)

    retrieved_doc = search_results[0].payload['doc_id']
    return answer, retrieved_doc