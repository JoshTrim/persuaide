from fastapi import FastAPI
from query import query_db, parse_response

app = FastAPI()

@app.get("/q={query}")
async def query_api(query):
    if query:
        res = query_db(query)
        res = parse_response(res)
        return {"result": res}

