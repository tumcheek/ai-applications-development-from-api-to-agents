import json

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/docs/guides/speech-to-text

#TODO:
# You need to transcribe 'audio_sample.mp3':
#   - Create Client that will go to transcriptions OpenAI API
#   - Call API and provide file (pay attention that you work with 'multipart/form-data')
#   - Get response with transcription
# ---
# Hints:
#   - Use /v1/audio/transcriptions endpoint
#   - Use whisper-1 or gpt-4o-transcribe model
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",


}

data = {
    "model": "gpt-transcribe",
}

file_name = "audio_sample.mp3"
file_path = f"t3_content_generation\\t3\{file_name}"

files = {
    "file": (file_name, open(file_path, "rb"), "audio/mpeg"),
}

endpoint = f'{OPENAI_HOST}/v1/audio/transcriptions'

response = requests.post(endpoint, data=data, headers=headers, files=files)

print(response.json())