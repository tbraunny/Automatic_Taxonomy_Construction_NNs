from typing import Union, Annotated
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the static directory to serve images and other static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("taxon_neuro.html", encoding="utf-8") as f:
        return f.read()
    
@app.get("/about.html", response_class=HTMLResponse)
def read_about():
    with open("about.html", encoding="utf-8") as f:
        return f.read()
    
@app.get("/gen_ont.html", response_class=HTMLResponse)
def read_about():
    with open("gen_ont.html", encoding="utf-8") as f:
        return f.read()
    
@app.get("/gen_tax.html", response_class=HTMLResponse)
def read_about():
    with open("gen_tax.html", encoding="utf-8") as f:
        return f.read()

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
