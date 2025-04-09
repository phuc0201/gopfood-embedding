from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

model = SentenceTransformer('intfloat/multilingual-e5-base')

@app.post("/embed")
async def embed(request: Request):
    body = await request.json()
    text = body.get("text", "")
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
