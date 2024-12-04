from typing import Union
from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from src.rag import tree_prompting  # Import your tree_prompting module
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Mount the static directory to serve images and other static files
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../front_end/static"))
template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../front_end/templates"))

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

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_price": item.price, "item_id": item_id}

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

# process user input for chats with Llama
@app.post("/process_input/")
async def process_input(user_input: str):
    try:
        #result = 
        result = tree_prompting.process_input(user_input)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
