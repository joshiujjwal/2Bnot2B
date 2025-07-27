from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from rag import RAG
import os
import aiofiles
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG system
rag = RAG()

# Create directories
os.makedirs("data/uploads", exist_ok=True)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,

    })

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        allowed_extensions = {'.txt'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            return JSONResponse({
                "status": "error",
                "message": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            })
        
        # Save uploaded file
        file_path = f"data/uploads/{file.filename}"
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process document
        success = rag.add_document(file_path, file.filename)
        
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"Successfully processed {file.filename}"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "Failed to process the document"
            })
            
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Error uploading file: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)