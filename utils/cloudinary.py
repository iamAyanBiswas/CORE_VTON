import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()

config=cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)


async def upload_image_to_cloudinary(image_bytes):
    # Upload the image to Cloudinary
    if not image_bytes:
        raise ValueError("Image bytes cannot be None")
    try:
        response = cloudinary.uploader.upload(image_bytes)
        return response["secure_url"]
    except Exception as e:
        print(e)
        raise e
