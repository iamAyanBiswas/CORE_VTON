from dotenv import load_dotenv
import os
from pyngrok import ngrok


load_dotenv()
NGROK_AUTH_KEY=os.getenv("NGROK_AUTH_KEY")


def get_ngrok_url(port=8000):
  ngrok.set_auth_token(NGROK_AUTH_KEY)
  public_url =  ngrok.connect(port).public_url
  print(f"Public URL: {public_url}")
  
