# redis.py
import os
import redis
from dotenv import load_dotenv

load_dotenv()


r = redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=True
)

QUEUE_NAME = os.getenv("VTON_QUEUE")
