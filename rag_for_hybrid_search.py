from nltk.corpus.reader import chunked
from pinecone import Pinecone
import os
import re
from uuid import uuid4
from typing import IO, Any, Dict, List, Tuple
from copy import deepcopy
import requests

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
import openai
from pinecone.core.client.model.query_response import QueryResponse
from pinecone import ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from dotenv import load_dotenv

from pathlib import Path

import itertools
import pickle

load_dotenv()

# Auth
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Test
assert len(PINECONE_API_KEY) > 0

# Pickle utils
def dump_pickle(dictionary: dict):
    file_name = list(dictionary.keys())[0]

    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(dictionary, f)

def restore_pickle(file_name: str):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def generate_chunks(doc: str, chunk_size: int = 512, chunk_overlap: int = 35) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.create_documents(doc)

def chunk_documents(docs: Dict[str, List[Text]], chunk_size: int = 512, chunk_overlap:int = 35) -> None:
    for key, value in docs.items():
        chunks = generate_chunks(value["content"])
        docs[key]["chunks"] = [c.page_content for c in chunks]

def produce_embeddings(chunks: List[str]) -> List[str]:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings = []
    for c in chunks:
        embedding = model.encode(c)
        embeddings.append(embedding)
    return embeddings

def create_ids(chunks: str) -> List[str]:
    return [str(uuid4()) for _ in range(len(chunks))]

def create_metadata_objs(doc: List[str], author: str, title: str) -> List[dict[str]]:
    return [
        {
        'text': d,
        'author': author,
        'title': title
        }
    for d in doc]

def create_composite_objects(texts):

    to_index = []

    for k, v in texts.items():
        for i in range(len(v["metadatas"])):
            to_index_obj = {
                'id': v["ids"][i],
                # 'sparse_values': v["sembeddings"][i],
                'values': v["dembeddings"][i],
                'metadata': v["metadatas"][i]
            }
            to_index.append(to_index_obj)
    return to_index

# Data
data = Path("./data/extracted/")

texts = {}
for file in data.glob("**/*.txt"):

    # read in content
    with open(file, "r") as f:
        text = f.readlines()

    # parse title and author
    author = file.stem.split('-')[0]
    title = file.stem.split('-')[1].split('(')[0][1:-1]

    texts.update({
        title: {
            "content": text,
            "author": author,
            "title": title
        }
    })

# uncomment to regenerate

# # chunk
chunk_documents(texts)
chunked_files = texts

# # dense embeddings
# for text in list(texts.keys()):
#     embeddings = produce_embeddings(chunked_files.get(text))
#     pkl = {text: embeddings}
#     dump_pickle(pkl)

for text in list(texts.keys()):
    data = restore_pickle(text + ".pkl")
    texts[text].update({"dembeddings": data[text]})

# confirm shape of dense embeddings
print(texts[list(texts.keys())[0]].keys())

for key, value in texts.items():
    assertion_value = [0 for i in value["dembeddings"] if i.shape == 384]
    assert sum(assertion_value) == 0

# sparse embeddings

# join all texts into 1 large corpus to calculate vocab
corpus = ""

for k, v in chunked_files.items():
    corpus += " ".join(v["chunks"])

print(len(corpus))

bm25 = BM25Encoder()
bm25.fit(corpus)

for k, v in texts.items():
    sparse_embed = [bm25.encode_documents(i) for i in v["chunks"]]
    texts[k].update({"sembeddings": sparse_embed})
    dump_pickle({f"{k}-sparse": sparse_embed})
    # data = restore_pickle(f"{k}-sparse.pkl")
    # sparse_embeddings[k] = data[k]

for k, v in list(texts.items()):
    assert len(texts[k]["sembeddings"]) == len(texts[k]["chunks"])
    print(len(texts[k]["sembeddings"]))

# pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "persuaide-4th-july"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        spec = ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension = 384,
        metric = "dotproduct",
    )

index_details = pc.describe_index(index_name)
index = pc.Index(index_name)

for k, v in list(texts.items()):
    texts[k].update({"ids": create_ids(v["chunks"])})
    assert len(v["ids"]) == len(v["chunks"])

for k, v in list(texts.items()):
    texts[k].update({"metadatas": create_metadata_objs(v["chunks"], v["author"], v["title"])})

objs_to_upsert = create_composite_objects(texts)

# for obj in objs_to_upsert:
#     for k, v in obj["sparse_values"].items():
#         print(k, len(v))

index.describe_index_stats()

print(len(objs_to_upsert))
# print(objs_to_upsert[0])



def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


# objs_generator = (x for x in objs_to_upsert)
#
#
# print("> Starting upsert...")
# # Upsert data with 100 vectors per upsert request
# for ids_vectors_chunk in chunks(objs_generator, batch_size=100):
#     print("> > Upserting chunk")
#     print(ids_vectors_chunk)
#     # index.upsert(vectors=ids_vectors_chunk)

from tqdm.auto import tqdm  # for progress bar

batch_size = 100
print("> Starting upsert...")
for i in tqdm(range(0, len(objs_to_upsert), batch_size)):
    i_end = min(len(objs_to_upsert), i+batch_size)

    # get batch of data
    batch = objs_to_upsert[i:i_end]
    
    # print("> > Upserting batch")
    # print(batch)

    # add to Pinecone
    index.upsert(vectors=batch)

