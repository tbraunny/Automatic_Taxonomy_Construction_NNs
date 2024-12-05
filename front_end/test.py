from typing import Union
from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from src.rag import tree_prompting  # Import your tree_prompting module
from utils.rag_engine import LocalRagEngine
from fastapi.templating import Jinja2Templates
import shutil

app = FastAPI()

# Mount the static directory to serve images and other static files
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../front_end/static"))
template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../front_end/templates"))
upload_path = os.path.abspath(os.path.join(os.path.dirname(__file__) , "../data/raw"))
cw_paper = 0

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print(static_path)
app.mount("/static", StaticFiles(directory=static_path), name="static")

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

templates = Jinja2Templates(directory=template_path)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    #base_path = os.path.dirname(__file__)  # Directory of the current script
    #file_path = os.path.join(base_path, "taxon_neuro.html")    
    #with open(file_path, encoding="utf-8") as f:
       # return f.read()
    return templates.TemplateResponse(
        request=request, name='taxon_neuro.html'
    )
    
@app.get("/about", response_class=HTMLResponse)
def read_about(request: Request):
    # file_path = os.path.join(base_path, "about.html")
    # with open("about.html", encoding="utf-8") as f:
    #     return f.read()

    return templates.TemplateResponse(
        request=request, name='about.html'
    )
    
@app.get("/gen_ont", response_class=HTMLResponse)
def read_gen_ont(request: Request):
    # with open("gen_ont.html", encoding="utf-8") as f:
    #     return f.read()
    return templates.TemplateResponse(
        request=request, name='gen_ont.html'
    )
    
@app.get("/gen_tax", response_class=HTMLResponse)
def read_gen_tax(request: Request):
    # with open("gen_tax.html", encoding="utf-8") as f:
    #     return f.read()
    return templates.TemplateResponse(
        request=request, name='gen_tax.html'
    )

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    #cw_paper = os.path.join(upload_path , file.filename)
    unique_file = unique_file_name(upload_path , file.filename)
    cw_paper = os.path.join(upload_path , unique_file)

    try:
        with open(cw_paper , "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally: # prevent semaphore warnings
        await file.close()
    
    return {"filename": unique_file, "message": "File uploaded successfully!"}

# check for unique file name, ensure new files do not overwrite old files
def unique_file_name(path , file_name):
    base_name , ext = os.path.splitext(file_name)
    new_file_name = file_name
    file_suffix = 1

    while (os.path.exists(os.path.join(path , new_file_name))):
        new_file_name = f"{base_name}_({file_suffix}){ext}"
        file_suffix += 1

    return new_file_name

# process user input for chats with Llama
@app.post("/process_input/")
async def process_input(request: Request):
    try:
        data = await request.json()
        user_input = data.get("prompt")
        if not user_input:
            return JSONResponse(content={"error": "No prompt provided"}, status_code=400)
        
        # Call the function from tree_prompting.py to process the input
        query_engine = LocalRagEngine(pdf_path=cw_paper , llm_model='llama3.2:3b-instruct-fp16').get_rag_engine()
        response = query_engine.query(user_input)
        print("response type " , type(response))
        print("response" , response)
        print(response)

        response_text = str(response)

        return JSONResponse(content={"response": response_text})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)