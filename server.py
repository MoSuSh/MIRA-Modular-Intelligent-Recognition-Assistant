from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import json
from database import insert_memory, get_all_memories

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

@app.get("/memories")
async def read_memories():
    raw_memories = get_all_memories()
    formatted_memories = []
    
    for name, embedding_str in raw_memories:
        try:
            embedding_list = json.loads(embedding_str)
            formatted_memories.append({
                "name": name,
                "embedding": embedding_list
            })
        except Exception:
            continue
            
    return formatted_memories