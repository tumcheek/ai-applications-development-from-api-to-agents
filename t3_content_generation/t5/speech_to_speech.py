import base64
import json
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/docs/guides/audio#add-audio-to-your-existing-application

#TODO:
# You need to generate answer in audio format based on the audio message:
#   - Create Client that is similar with OpenAIClients but extracts from message audio (instead of content)
#   - Call API
#   - Get response as base64 content, decode and save as .mp3 file
# ---
# Hints:
#   - Use /v1/chat/completions endpoint
#   - Use gpt-4o-audio-preview model
#   - Use modalities=["text", "audio"]
#   - Use audio={"voice": "ballad", "format": "mp3"}
#   - Use similar method to encode audio as you have done for images encoding


def encode_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

file_path = "t3_content_generation\\t5"
audio_base64 = encode_file(f"{file_path}\question.mp3")
audio_base64_url = f"data:audio/mpeg;base64,{audio_base64}"

endpoint = f"{OPENAI_HOST}/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-type": "application/json"

}

data = json.dumps({
    "model": "gpt-audio-1.5",
    "modalities": ["text", "audio"],
    "audio": { "voice": "alloy", "format": "mp3" },
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Please respond to this voice message." },
          {
            "type": "input_audio",
            "input_audio": {
              "data": audio_base64_url,
              "format": "mp3"
            }
          }
        ]
      }
    ]
  }
)


response = requests.post(endpoint, data=data, headers=headers)

response_json = response.json()
audio_data = response_json["choices"][0]["message"]["audio"]["data"]
output_file = f"{file_path}/f{response_json.get("created")}_answer.mp3"
with open(output_file, "wb") as f:
    f.write(base64.b64decode(audio_data))

