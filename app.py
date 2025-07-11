from flask import Flask, request
import openai
import os
import tempfile
import base64
import logging
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client as TwilioClient
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize Flask app
app = Flask(__name__)

# Load OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Logging
logging.basicConfig(level=logging.INFO)

# Load RAG content
with open("arslanasghar_full_content.txt", "r", encoding="utf-8") as f:
    full_text = f.read()
rag_chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]

# Load embedding model (lightweight)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
torch.set_grad_enabled(False)

# Audio buffers
buffers = {}

@app.route("/ws-audio", methods=["POST"])
def websocket_audio():
    data = request.get_json()
    call_sid = data.get("streamSid")

    if data.get("event") == "media":
        payload = base64.b64decode(data["media"]["payload"])
        buffers.setdefault(call_sid, b"").__iadd__(payload)

        if len(buffers[call_sid]) > 16000 * 5:  # ~5 seconds audio
            audio_bytes = buffers[call_sid]
            buffers[call_sid] = b""

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                audio_path = f.name

            with open(audio_path, "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)
            user_text = transcript["text"].strip()

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
    query_embedding = embedder.encode(user_text, convert_to_tensor=True, device='cpu')

    top_chunks = []
    for chunk in rag_chunks:
        chunk_embedding = embedder.encode(chunk, convert_to_tensor=True, device='cpu')
        similarity = util.pytorch_cos_sim(query_embedding, chunk_embedding).item()

        if len(top_chunks) < 2:
            top_chunks.append((similarity, chunk))
        else:
            min_sim = min(top_chunks, key=lambda x: x[0])[0]
            if similarity > min_sim:
                top_chunks.sort(key=lambda x: x[0])
                top_chunks[0] = (similarity, chunk)

    context = "\n".join([chunk for _, chunk in sorted(top_chunks, reverse=True)])

    messages = [
        {"role": "system", "content": (
            "You are Arslan, a professional digital marketer from Doha. "
            "Speak concisely and naturally, as if you are human. Use only relevant info from the context."
        )},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_text}"}
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
