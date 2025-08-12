# !pip install fastapi uvicorn py-localtunnel
# !mkdir api
# %cd api
# !touch main.py
# !curl ipecho.net/plain
# !uvicorn main:app & pylt port 8000


# !git clone -b dev https://github.com/iamAyanBiswas/CORE_VTON
# %cd CORE_VTON
# !pip install -r requirements.txt
# !python main.py


# ex-payload
# {
#   "person_image_url": "https://zhengchong-catvton.hf.space/file=/tmp/gradio/ba5ba7978e7302e8ab5eb733cc7221394c4e6faf/model_5.png",
#   "cloth_image_url": "https://zhengchong-catvton.hf.space/file=/tmp/gradio/863e42db021d45ae5281225735e64c15996d9f62/23255574_53383833_1000.jpg",
#   "cloth_type": "upper",
#   "num_inference_steps": 10,
#   "guidance_scale": 3,
#   "seed": 42,
#   "show_type": "result only"
# }

import io
from PIL import Image


import asyncio
from pydantic import BaseModel
from validators import url

import aiohttp
import tempfile
import os

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse


from utils.api_response import ApiResponse
# from vton_model.app import vton
from vton_model.app import vton
from utils.cloudinary import upload_image_to_cloudinary    
from utils.ngrok import get_ngrok_url

class VTonRequest(BaseModel):
    person_image_url: str
    cloth_image_url: str
    cloth_type: str
    num_inference_steps: int
    guidance_scale: float
    seed: int
    show_type: str


app = FastAPI()

get_ngrok_url(8000)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail if isinstance(exc.detail, str) else exc.detail.get("message")},
    )




async def pil_image_to_bytes_async(image: Image.Image, format="PNG") -> bytes:
    def sync_save():
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()
    return await asyncio.to_thread(sync_save)   




async def download_image_to_temp_async(url):
    # Get file extension or default to .jpg
    ext = os.path.splitext(url)[-1]
    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        ext = '.jpg'

    # Create a NamedTemporaryFile (closed so aiohttp can write)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_file.close()

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                with open(temp_file.name, 'wb') as f:
                    while True:
                        chunk = await resp.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                return temp_file.name
            else:
                raise Exception(f"Failed to download image, status code: {resp.status}")



@app.get("/")
async def root(query=""):
    return ApiResponse(message="Welcome to the VTON API", data={"query": query}).json()

@app.post("/vton")
async def vton_api(request:VTonRequest):
    person_image_url= request.person_image_url
    cloth_image_url= request.cloth_image_url
    cloth_type= request.cloth_type
    num_inference_steps= request.num_inference_steps
    guidance_scale= request.guidance_scale
    seed= request.seed
    show_type= request.show_type

    required_fields = [
        "person_image_url",
        "cloth_image_url",
        "cloth_type",
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "show_type",
    ]

    # Check for missing fields
    for field in required_fields:
        if getattr(request, field, None) in (None, ""):
            raise HTTPException(status_code=400, detail=f"{field} is required")

    # Validate URLs
    if not url(person_image_url):
        raise HTTPException(status_code=400, detail="person_image_url is not valid")
    if not url(cloth_image_url):
        raise HTTPException(status_code=400, detail="cloth_image_url is not valid")

    # Validate cloth_type
    if not cloth_type in ["upper", "lower", "overall"]:
        raise HTTPException(status_code=400, detail="cloth_type must be one of 'upper', 'lower', 'overall'")

    # Validate num_inference_steps
    if not 10<=num_inference_steps<=100:
        raise HTTPException(status_code=400, detail="num_inference_steps must be between 10 and 100")

    # Validate guidance_scale
    if not 0.1<=guidance_scale<=7.5:
        raise HTTPException(status_code=400, detail="guidance_scale must be between 0.1 and 7.5")

    # Validate seed
    if not -1<=seed<=1000:
        raise HTTPException(status_code=400, detail="seed must be between -1 and 1000")

    # Validate show_type
    if not show_type in ["result only", "input & result", "input & mask & result"]:
        raise HTTPException(status_code=400, detail="show_type must be one of 'result only', 'input & result', 'input & mask & result'")

    # Download images
    try:
        person_bytes= await download_image_to_temp_async(person_image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unexpected error downloading person_image_url")
    try:
        cloth_bytes= await download_image_to_temp_async(cloth_image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unexpected error downloading cloth_image_url")

    # Run blocking GPU function in executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    try:
        generated_image = await loop.run_in_executor(
            None,
            vton,
            person_bytes,
            cloth_bytes,
            cloth_type,
            num_inference_steps,
            guidance_scale,
            seed,
            show_type,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Unexpected error occurred during generating image")

    try:
      buffer = io.BytesIO()
      await loop.run_in_executor(None, generated_image.save, buffer, "PNG")
      buffer.seek(0)
      generated_image_bytes=buffer.read()

    except Exception as e:
      raise HTTPException(status_code=400, detail="Unexpected error occurred during transform image putput")
    try:
        output_url = await upload_image_to_cloudinary(generated_image_bytes)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Unexpected error occurred during uploading image")


    return ApiResponse(data={"url": output_url}).json()




