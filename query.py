from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from typing import List, Any, Dict, Tuple
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone.core.client.model.query_response import QueryResponse

load_dotenv()


# Auth
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def produce_embeddings(chunks: List[str]) -> List[str]:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings = []
    for c in chunks:
        embedding = model.encode(c).tolist()
        embeddings.append(embedding)
    return embeddings

# pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "persuaide-4th-july"

index_details = pc.describe_index(index_name)
index = pc.Index(index_name)

print(index_details)

def query_db(query="inflation is a tax on the poor"):
    query_dembeddings = produce_embeddings([query])
    result = index.query(vector=query_dembeddings[0], top_k=10, include_metadata=True)
    return result

def parse_response(response):
    parsed_response = []    
    for row in response["matches"]:
        parsed_row = {
            "author": row["metadata"]["author"],
            "title": row["metadata"]["title"],
            "text": row["metadata"]["text"],
        }
        parsed_response.append(parsed_row)

    return parsed_response
