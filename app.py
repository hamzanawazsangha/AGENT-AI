from flask import Flask, request
import openai
import whisper
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

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Logging
logging.basicConfig(level=logging.INFO)

# Load Whisper model once
stt = whisper.load_model("tiny", device="cpu")

# Load Arslan's knowledge base
with open("arslanasghar_full_content.txt", "r", encoding="utf-8") as f:
    full_text = f.read()
rag_chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]

# Load SentenceTransformer and compute embeddings only once
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(rag_chunks, convert_to_tensor=True, device='cpu')

# Audio buffers
buffers = {}

@app.route("/ws-audio", methods=["POST"])
def websocket_audio():
    data = request.get_json()
    call_sid = data.get("streamSid")

    if data.get("event") == "media":
        payload = base64.b64decode(data["media"]["payload"])
        buffers.setdefault(call_sid, b"").__iadd__(payload)

        if len(buffers[call_sid]) > 16000 * 5:  # 5 sec audio
            audio_bytes = buffers.pop(call_sid, b"")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                audio_path = f.name

            try:
                audio_tensor = whisper.load_audio(audio_path)
                audio_tensor = whisper.pad_or_trim(audio_tensor)
                mel = whisper.log_mel_spectrogram(audio_tensor).to("cpu")

                result = whisper.decode(stt, mel, whisper.DecodingOptions(fp16=False))
                user_text = result.text.strip()
                logging.info(f"[{call_sid}] said: {user_text}")

                if user_text:
                    reply = get_ai_reply(user_text)
                    logging.info(f"Arslan replies: {reply}")
                    say_to_caller(call_sid, reply)

            except Exception as e:
                logging.error(f"Error processing audio: {e}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

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

    logging.info(f"Started streaming for: {from_number}")
    return str(response)


def get_ai_reply(user_text):
    # Embed user query and search
    query_embedding = embedder.encode(user_text, convert_to_tensor=True, device='cpu')
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=2)
    top_chunks = [rag_chunks[hit["corpus_id"]] for hit in hits[0]]
    context = "\n".join(top_chunks)

    # Ask OpenAI using the context
    messages = [
        {"role": "system", "content": (
            "You are Arslan, a helpful and professional digital marketer based in Doha. "
            "Answer briefly, like a human, using only relevant information from the context."
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
