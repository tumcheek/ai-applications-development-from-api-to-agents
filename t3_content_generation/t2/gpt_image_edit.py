import base64
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/reference/resources/images/methods/edit
# ---
# Request (multipart/form-data, NOT json):
# curl -X POST "https://api.openai.com/v1/images/edits" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -F "model=gpt-image-1" \
#     -F "image=@logo.png" \
#     -F "prompt=Add magical sparkles and glowing aura around the logo"
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": "Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO..."
#     }
#   ]
# }

#TODO:
# You need to edit an existing image with `gpt-image-2` model:
#   - Take a local image (e.g. 'logo.png') and a prompt describing the edit
#   - Send it to the OpenAI images edit API
#   - Decode the returned base64 image and save it locally
# ---
# Hints:
#   - Use /v1/images/edits endpoint
#   - The request must be 'multipart/form-data' (NOT json) — pass the image as a file and the prompt/model as form fields
#   - The edited image will be returned in base64 format

endpoint = f"{OPENAI_HOST}/v1/images/edits"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",

}
file_name = "logo.png"
file_path = f"t3_content_generation\\t2\{file_name}"
files = {
    "image[]": (file_name, open(file_path, "rb"), "image/png"),
}

data = {
    "model": "gpt-image-1.5",
    "prompt": "Add magical sparkles and glowing aura around the logo",
}
response = requests.post(endpoint,headers=headers, data=data, files=files)
payload = response.json()
response.raise_for_status()
new_name = f"edit_{file_name}_{payload.get("created")}.png"
new_file_path = f"t3_content_generation\\t2\{new_name}"
image_base64 = payload["data"][0]["b64_json"]
image_bytes = base64.b64decode(image_base64)

with open(new_file_path, "wb") as f:
    f.write(image_bytes)