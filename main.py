# main.py
import json
import os
import tempfile
import io
import requests
from validators import url
from utils.redis import r, QUEUE_NAME
from utils.postgresql import update_job_status
from utils.cloudinary import upload_image_to_cloudinary
from vton_model.app import vton

REQUIRED_FIELDS = [
    "id",
    "person_image_url",
    "cloth_image_url",
    "cloth_type",
    "num_inference_steps",
    "guidance_scale",
    "seed",
    "show_type",
]

CLOTH_TYPES = ["upper", "lower", "overall"]
SHOW_TYPES = ["result only", "input & result", "input & mask & result"]

# Synchronous image download (returns temp file path)
def download_image_to_temp(url_str):
    ext = os.path.splitext(url_str)[-1]
    if ext.lower() not in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
        ext = ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        resp = requests.get(url_str, stream=True, timeout=30)
        resp.raise_for_status()
        with open(temp_file.name, "wb") as f:
            for chunk in resp.iter_content(1024):
                if not chunk:
                    break
                f.write(chunk)
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise e

def validate_job(job_dict):
    for field in REQUIRED_FIELDS:
        if field not in job_dict or job_dict[field] in (None, ""):
            raise ValueError(f"{field} is required")
    if not url(job_dict["person_image_url"]):
        raise ValueError("person_image_url is not valid")
    if not url(job_dict["cloth_image_url"]):
        raise ValueError("cloth_image_url is not valid")
    if job_dict["cloth_type"] not in CLOTH_TYPES:
        raise ValueError("cloth_type must be one of 'upper', 'lower', 'overall'")
    if not (10 <= int(job_dict["num_inference_steps"]) <= 100):
        raise ValueError("num_inference_steps must be between 10 and 100")
    if not (0.1 <= float(job_dict["guidance_scale"]) <= 7.5):
        raise ValueError("guidance_scale must be between 0.1 and 7.5")
    if not (-1 <= int(job_dict["seed"]) <= 1000):
        raise ValueError("seed must be between -1 and 1000")
    if job_dict["show_type"] not in SHOW_TYPES:
        raise ValueError("show_type must be one of 'result only', 'input & result', 'input & mask & result'")

def worker_loop():
    while True:
        try:
            job_data = r.blpop(QUEUE_NAME, timeout=0)
            if not job_data:
                print("Empty Queue")
                continue
            _, raw = job_data
            job_dict = json.loads(raw)
            job_id = job_dict.get("id")
            print(f"Processing job {job_id} ...")
            try:
                validate_job(job_dict)
            except Exception as e:
                print(f"Validation error: {e}")
                if job_id:
                    update_job_status(job_id, "failed")
                continue
            update_job_status(job_id, "processing")

            # Download images
            try:
                person_path = download_image_to_temp(job_dict["person_image_url"])
            except Exception as e:
                print(f"Error downloading person_image_url: {e}")
                update_job_status(job_id, "failed")
                continue
            try:
                cloth_path = download_image_to_temp(job_dict["cloth_image_url"])
            except Exception as e:
                print(f"Error downloading cloth_image_url: {e}")
                update_job_status(job_id, "failed")
                os.unlink(person_path)
                continue

            # Run VTON model
            try:
                result_image = vton(
                    person_path,
                    cloth_path,
                    job_dict["cloth_type"],
                    int(job_dict["num_inference_steps"]),
                    float(job_dict["guidance_scale"]),
                    int(job_dict["seed"]),
                    job_dict["show_type"]
                )
            except Exception as e:
                print(f"Error running VTON model: {e}")
                update_job_status(job_id, "failed")
                os.unlink(person_path)
                os.unlink(cloth_path)
                continue

            # Save result image to bytes
            buffer = io.BytesIO()
            try:
                result_image.save(buffer, format="PNG")
                buffer.seek(0)
                image_bytes = buffer.read()
            except Exception as e:
                print(f"Error saving result image: {e}")
                update_job_status(job_id, "failed")
                os.unlink(person_path)
                os.unlink(cloth_path)
                continue
            try:
                image_url = upload_image_to_cloudinary(image_bytes)
            except Exception as e:
                print(f"Error uploading to Cloudinary: {e}")
                update_job_status(job_id, "failed")
                os.unlink(person_path)
                os.unlink(cloth_path)
                continue
            # Update job status
            update_job_status(job_id, "completed", image_url=image_url, update=True)
            print(f"✅ Job {job_id} completed: {image_url}")
            # Cleanup
            os.unlink(person_path)
            os.unlink(cloth_path)
        except Exception as e:
            print(f"❌ Worker error: {e}")
            # Optionally: add retry logic or DLQ

if __name__ == "__main__":
    worker_loop()
