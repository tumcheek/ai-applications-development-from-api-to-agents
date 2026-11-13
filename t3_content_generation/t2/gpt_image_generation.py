import base64
from datetime import datetime
import json

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/reference/resources/images/methods/generate
# ---
# Request:
# curl -X POST "https://api.openai.com/v1/images/generations" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -H "Content-type: application/json" \
#     -d '{
#         "model": "gpt-image-2",
#         "prompt": "smiling catdog."
#     }'
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO...,
#     }
#   ]
# }

#TODO:
# You need to create some images with `gpt-image-2` model:
#   - Generate an image with 'Smiling catdog'
#   - Decode and save it locally
# ---
# Hints:
#   - Use OpenAIClientT3 to connect to OpenAI API
#   - Use /v1/images/generations endpoint
#   - The image will be returned in base64 format
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-type": "application/json"

}

data = json.dumps({
    "model": "gpt-image-2.5-flare",
    "prompt": "smiling catdog."
})

endpoint = f"{OPENAI_HOST}/v1/images/generations"

response = requests.post(endpoint,headers=headers, data=data)
payload = response.json()
response.raise_for_status()
new_name = f"new_catdog_{payload.get("created")}.png"
new_file_path = f"t3_content_generation\\t2\{new_name}"
image_base64 = payload["data"][0]["b64_json"]
image_bytes = base64.b64decode(image_base64)

with open(new_file_path, "wb") as f:
    f.write(image_bytes)