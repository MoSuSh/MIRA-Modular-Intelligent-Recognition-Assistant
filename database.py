import json
import sqlite3
from sqlite3 import Error
from typing import Any, Dict, List, Optional

DB_FILE = "mira.db"


def get_db_connection() -> Optional[sqlite3.Connection]:
    """Establishes and returns a database connection with dictionary-style row access."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None


def init_db() -> None:
    """Initializes the database and creates the memories table if it doesn't exist."""
    conn = get_db_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            print("Database initialized and verified successfully.")
        except Error as e:
            print(f"Failed to initialize database: {e}")
        finally:
            conn.close()


def insert_memory(name: str, embedding: List[float]) -> bool:
    """Serializes floating-point embedding vector to JSON string and stores it in the DB."""
    conn = get_db_connection()
    if conn is not None:
        try:
            embedding_str = json.dumps(embedding)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (name, embedding)
                VALUES (?, ?);
                """,
                (name, embedding_str),
            )
            conn.commit()
            return True
        except Error as e:
            print(f"Memory insertion failed: {e}")
            return False
        finally:
            conn.close()
    return False


def get_all_memories() -> List[Dict[str, Any]]:
    """Retrieves all stored memories and converts JSON strings back into float lists."""
    conn = get_db_connection()
    memories = []
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, embedding, created_at FROM memories ORDER BY created_at DESC;"
            )
            rows = cursor.fetchall()

            for row in rows:
                memories.append({
                    "id": row["id"],
                    "name": row["name"],
                    "embedding": json.loads(row["embedding"]),
                    "created_at": row["created_at"],
                })
        except (Error, json.JSONDecodeError) as e:
            print(f"Failed to fetch memories: {e}")
        finally:
            conn.close()
    return memories


if __name__ == "__main__":
    init_db()
