import json
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


class Voice:
    alloy: str = 'alloy'
    ash: str = 'ash'
    ballad: str = 'ballad'
    coral: str = 'coral'
    echo: str = 'echo'
    fable: str = 'fable'
    nova: str = 'nova'
    onyx: str = 'onyx'
    sage: str = 'sage'
    shimmer: str = 'shimmer'


# https://developers.openai.com/api/docs/guides/text-to-speech
# Request:
# curl https://api.openai.com/v1/audio/speech \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "gpt-4o-mini-tts",
#     "input": "Why can't we say that black is white?",
#     "voice": "coral",
#     "instructions": "Speak in a cheerful and positive tone."
#   }' \
# Response:
#   bytes with audio

#TODO:
# You need to convert text to speech:
#   - Create Client that will go to speech OpenAI API
#   - Call API
#   - Get response and save as .mp3 file
# ---
# Hints:
#   - Use /v1/audio/speech endpoint
#   - Use gpt-4o-mini-tts model


headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-type": "application/json"

}

data = json.dumps({
    "model": "gpt-4o-mini-tts",
    "input": "Why can't we say that black is white?",
    "voice": Voice.coral,
    "instructions": "Speak in a cheerful and positive tone."
  }
)

endpoint = f'{OPENAI_HOST}/v1/audio/speech'

response = requests.post(endpoint, data=data, headers=headers)

response.raise_for_status()
name = f"answer.mp3"
file_path = f"t3_content_generation\\t4\{name}"

with open(file_path, "wb") as f:
    f.write(response.content)