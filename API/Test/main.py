#from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/entrypoint")
def read():
    return {"Hello":"World"}