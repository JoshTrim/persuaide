from fastapi import FastAPI, Body
from query import query_db, parse_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:5173",  # Vite dev frontend origin
    "http://localhost:4173",  # Vite preview frontend origin
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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