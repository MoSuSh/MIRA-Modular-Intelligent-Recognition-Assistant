import sqlite3
from sqlite3 import Error

DB_FILE = "mira.db"

def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        return conn
    except Error as e:
        print(f"Database error: {e}")
        return None 
    
def init_db():
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
            print("Database initialized and table created successfully!")
        except Error as e:
            print(f"Failed to initialize and/or create table: {e}")
        finally:
            conn.close()

def insert_memory(name: str, embedding_str: str):
    conn = get_db_connection()
    
    if conn is not None:
        try:
            cursor = conn.cursor()

            cursor.execute("""
                           INSERT INTO memories (name, embedding)
                           VALUES(?, ?);
                           """, (name, embedding_str)
                           )
            conn.commit()
            return True
        
        except Error as e:
            print(f"Memory insertion failed: {e}")
            return False        
        finally:
            conn.close()
    return False
    
        
if __name__ == "__main__":
    init_db()