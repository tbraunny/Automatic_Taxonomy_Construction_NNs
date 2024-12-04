from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess

app = FastAPI()

class UserInput(BaseModel):
    input_text: str  # Define the expected structure of user input

@app.post("/process_input")
async def process_input(user_input: UserInput):
    try:
        # Call the tree_prompting.py script with user input as an argument
        result = subprocess.run(
            ["python3", "src/rag/tree_prompting.py", user_input.input_text],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Error running the Python script")

        # Return the result from the Python script
        return JSONResponse(content={"output": result.stdout.strip()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
