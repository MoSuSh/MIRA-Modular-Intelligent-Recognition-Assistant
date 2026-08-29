from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import get_all_memories, init_db, insert_memory

app = FastAPI(
    title="MIRA Backend Server",
    description="REST API for storing and fetching facial and object recognition embeddings.",
    version="1.0.0",
)

# Enable CORS for local cross-origin API calls from scripts or frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Automatically ensures SQLite database tables exist when starting the server."""
    init_db()


class MemoryCreate(BaseModel):
    name: str = Field(..., description="Name/Label of the entity")
    embedding: List[float] = Field(
        ..., description="Feature embedding vector as a list of numbers"
    )


@app.get("/", tags=["Health Check"])
async def health_check() -> Dict[str, str]:
    return {"status": "online", "system": "MIRA Server"}


@app.post(
    "/memories", status_code=status.HTTP_201_CREATED, tags=["Memories"]
)
async def create_memory(memory: MemoryCreate) -> Dict[str, str]:
    if not memory.name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name parameter cannot be blank.",
        )

    success = insert_memory(memory.name, memory.embedding)
    if success:
        return {
            "status": "success",
            "message": f"Memory '{memory.name}' successfully recorded.",
        }
    
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Server error: Failed to commit memory to database disk.",
    )


@app.get("/memories", tags=["Memories"])
async def fetch_memories() -> Dict[str, Any]:
    memories = get_all_memories()
    return {
        "status": "success",
        "count": len(memories),
        "data": memories,
    }
