from typing import Union
from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from tests.rag import tree_prompting  # Import your tree_prompting module
from utils.rag_papers.rag_engine import LocalRagEngine
from src.ontology.visualization.graph_viz import process_ontology
from fastapi.templating import Jinja2Templates
import shutil

app = FastAPI()

# Mount the static directory to serve images and other static files
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../front_end/static"))
template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../front_end/templates"))
upload_path = os.path.abspath(os.path.join(os.path.dirname(__file__) , "../data/user_temp"))
default_path = os.path.abspath(os.path.join(os.path.dirname(__file__) , "../data/raw"))

cw_paper = 0

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
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

@app.post("/ont_vis/")
async def create_ont_vis():
    try:
        result = process_ontology()
        return result
    except Exception as e:
        print("Unable to visualize the ontology, error: " , e)

# check for unique file name, ensure new files do not overwrite old files
def unique_file_name(path , file_name):
    base_name , ext = os.path.splitext(file_name)
    new_file_name = file_name
    file_suffix = 1

    while (os.path.exists(os.path.join(path , new_file_name))):
        new_file_name = f"{base_name}_({file_suffix}){ext}"
        file_suffix += 1

    return new_file_name

# getter for fastAPI functions to obtain current working paper
def get_cw_paper():
    return cw_paper

# process user input for chats with Llama
@app.post("/process_input/")
async def process_input(request: Request):
    try:
        data = await request.json()
        user_input = data.get("prompt")
        if not user_input:
            return JSONResponse(content={"error": "No prompt provided"}, status_code=400)
        
        cw_paper = get_cw_paper()
        if (cw_paper == 0): # check if user has inputted a paper, if not default to AlexNet
            cw_paper = os.path.join(default_path , "AlexNet.pdf")
        
        # Call the function from tree_prompting.py to process the input
        # if code conversion invoked llama_model = qwen
        code_str ="all the code"
        instructions = f"""The text within the triple delimiters is a user-prompted query, asnwer the question like a good kitten. \
                           The following is the code, only refreence the following code, if query cannot be answered within the scope \
                           of the code then respond with I don't know. {code_str}"""
        instructions += '```{query}```'
        query = user_input

        #   response = OllamaModel(qwen).query_ollama(code , instructions)
        query_engine = LocalRagEngine(pdf_path=cw_paper , llm_model='llama3.2:3b-instruct-fp16').get_rag_engine()
        response = query_engine.query(user_input)

        response_text = str(response)

        return JSONResponse(content={"response": response_text})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)