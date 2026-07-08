from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import json
from database import insert_memory

app = FastAPI()

class MemoryCreate(BaseModel):
    name: str
    embedding: List[float]

@app.post("/memories")
async def create_memory(memory: MemoryCreate):

    embedding_string = json.dumps(memory.embedding)
    success = insert_memory(memory.name, embedding_string)
    if success:
        return {
            "status": "success",
            "message": f"Memory '{memory.name}' has been permanently saved to MIRA's database."
        }
    else:
        return {
            "status": "error",
            "message": "The server received the data, but failed to write it to disk."
        }
