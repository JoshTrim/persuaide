# persuaide

## Main files are:
**extract.py**: Convert any pdfs stored in /data/raw to .txt files saved to /data/extracted    
**rag_for_hybrid_search.py**: Initial implementation of chunking, embedding, and upserting to Pinecone db. Extracts author and title from .txt files and adds them into vector metadata. Upserts generated vectors (w/ ids and metadata) to Pinecone db.   
**query.py**: Helper to query db and parse response   
**api.py**: Localhost api to query db

## To use:
- Install python
- Create a virtualenv and install dependencies with `pip install -r requirements.txt`
- just api deps `pip install -r api_requirements.txt`


**The below two steps are only necessary if you are adding new content to the DB**
- Run extract.py to convert files from .pdf to .txt
- Run rag_for_hybrid_search.py to create vector embeddings and upsert to Pinecone db.


- Run `fastapi dev api.py` to run localhost api. Go to http://127.0.0.1:8000/docs to test API routes
