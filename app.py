from flask import Flask, request, jsonify
import openai
import whisper
import os
import tempfile
import requests
from twilio.twiml.voice_response import VoiceResponse, Start, Stream
from twilio.rest import Client as TwilioClient
import base64
import json
import logging

# Init Flask app
app = Flask(__name__)

# Load Whisper model and OpenAI client
stt = whisper.load_model("tiny", device="cpu")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Twilio credentials from Render environment
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Log basic info
logging.basicConfig(level=logging.INFO)

# WebSocket message buffer per call
buffers = {}

@app.route("/ws-audio", methods=["POST"])
def websocket_audio():
    data = request.get_json()
    call_sid = data.get("streamSid")

    if data.get("event") == "media":
        payload = base64.b64decode(data["media"]["payload"])
        buffers.setdefault(call_sid, b"").__iadd__(payload)

        if len(buffers[call_sid]) > 16000 * 5:
            audio_bytes = buffers[call_sid]
            buffers[call_sid] = b""

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                path = f.name

            audio_tensor = whisper.load_audio(path)
            audio_tensor = whisper.pad_or_trim(audio_tensor)
            mel = whisper.log_mel_spectrogram(audio_tensor).to(stt.device)
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(stt, mel, options)
            user_text = result.text.strip()

            logging.info(f"Caller [{call_sid}] said: {user_text}")

            if user_text:
                reply = get_ai_reply(user_text)
                logging.info(f"Arslan says: {reply}")
                say_to_caller(call_sid, reply)

    return ("", 204)


@app.route("/handle-call", methods=["POST"])
def handle_call():
    from_number = request.form.get("From")
    response = VoiceResponse()

    response.say("Hello, this is Arslan. I am listening.", voice="man")

    start = Start()
    stream_url = request.url_root.rstrip("/") + "/ws-audio"
    start.stream(url=stream_url)
    response.append(start)

    logging.info(f"Started streaming for caller: {from_number}")

    return str(response)


def get_ai_reply(user_text):
    messages = [
        {"role": "system", "content": "You are Arslan, a friendly digital marketer from Doha. Speak short and professionally."},
        {"role": "user", "content": user_text}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message["content"].strip()


def say_to_caller(call_sid, text):
    twilio_client.calls(call_sid).update(
        twiml=f'<Response><Say voice="man">{text}</Say></Response>'
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
