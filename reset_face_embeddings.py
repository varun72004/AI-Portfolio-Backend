"""
Reset face embeddings in MongoDB Atlas so the server re-seeds them
with the correct two-user setup (Face 1 -> Admin 1, Face 2 -> Admin 2).
Run this ONCE before starting the server.
"""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME", "test_database")

async def reset():
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    
    count = await db.face_embeddings.count_documents({})
    print(f"Found {count} existing face embeddings in MongoDB.")
    
    if count > 0:
        result = await db.face_embeddings.delete_many({})
        print(f"Deleted {result.deleted_count} face embeddings.")
        print("The server will re-seed them on next startup with the correct user mapping.")
    else:
        print("No embeddings to delete. Server will seed fresh on startup.")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(reset())
