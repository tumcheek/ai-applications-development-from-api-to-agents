import base64

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/docs/guides/images-vision?format=url&lang=curl
# https://developers.openai.com/api/docs/guides/images-vision?format=base64-encoded

#TODO:
# You need to analyse these 2 images:
#   - https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg
#   - in this folder we have 'logo.png', load it as encoded data (see documentation)
# ---
# Hints:
#   - Use OpenAIClientT3 to connect to OpenAI API
#   - Use /v1/chat/completions endpoint
#   - Function to encode image to base64 you can find in documentation
# ---
# In the end load both images (url and base64 encoded 'logo.png'), ask "Generate poem based on images" and se what will happen?

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAIClientT3(f"{OPENAI_HOST}/v1/chat/completions")
img_url = "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg"
img_base64 = encode_image("t3_content_generation\\t1\logo.png")
img__base64_url = f"data:image/jpeg;base64,{img_base64}"
data = {
  "model": "gpt-5.2",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Generate poem based on images"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": img_url
           }
        },
                {
          "type": "image_url",
          "image_url": {
            "url": img__base64_url
           }
        }
      ]
    }
  ]
}

ai_response = client.call(**data)

