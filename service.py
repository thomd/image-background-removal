from fastapi import FastAPI, UploadFile, File, Header
from fastapi.responses import RedirectResponse
import shutil
from typing import List, Optional
import inference
import uvicorn
import numpy as np

api = FastAPI()


@api.get("/")
def docs():
    return RedirectResponse(url="/docs")


@api.post("/image")
async def upload_img(file: UploadFile = File(...)):
    content = await file.read()
    base64_str = inference.remove_background(data=content)
    return {"image": f"data:image/png;base64,{base64_str}"}


@api.post("/file")
async def upload_image(file: UploadFile = File(...)):
    with open(f"upload/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    inference.remove_background(f"upload/{file.filename}")
    return {"file_name": file.filename}


@api.post("/files")
async def upload_images(files: List[UploadFile] = File(...)):
    for file in files:
        with open(f"upload/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"file_names": [f.filename for f in files]}


if __name__ == "__main__":
    uvicorn.run(api)
