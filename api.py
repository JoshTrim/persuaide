from fastapi import FastAPI, Body
from query import query_db, parse_response

app = FastAPI()

@app.get("/q={query}")
async def query_api(query):
    if query:
        res = query_db(query)
        res = parse_response(res)
        return {"result": res}

@app.post("/query")
async def query_api(query: str = Body(..., embed=True)):
    if query:
        res = query_db(query)
        res = parse_response(res)
        return {"result": res}