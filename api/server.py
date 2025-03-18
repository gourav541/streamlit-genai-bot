from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI()

# uvicorn server:app --host localhost --port 8000 --reload

# Folder where files are stored
directory_path = "../knowledge_base/MP_cooperative_societies/"

@app.get("/")
def home():
    return {"message": "FastAPI Server is running!"}

@app.get("/files/{filename}")
def get_file(filename: str):
    """Serve files from the knowledge base directory."""
    file_path = os.path.join(directory_path, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf")
    return {"error": "File not found"}
