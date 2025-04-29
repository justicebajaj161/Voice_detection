import os
import re
import io
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import webrtcvad
import speech_recognition as sr
from collections import deque

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
google_recognizer = sr.Recognizer()

# Constants
WAKE_WORDS = ["aditya", "hi aditya", "hey aditya", "hi aditya gupta", "hey aditya gupta"]
SAMPLE_RATE = 16000
CHUNK_SIZE = 480  # 30ms chunks
VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT = 2.0  # seconds of silence before stopping
MIN_UTTERANCE_LENGTH = 1.0  # seconds minimum speech
SPEECH_BUFFER_SECONDS = 0.5  # buffer before/after speech
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
INTRODUCTION = "Hello, this is Aditya Gupta. How can I assist you today?"

# Initialize VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def process_data_folder(data_folder="data"):
    """Process documents into ChromaDB with OpenAI embeddings"""
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    
    documents, metadatas, ids = [], [], []
    
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith((".txt", ".md")):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    documents.append(text)
                    metadatas.append({"source": path})
                    ids.append(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
    
    if documents:
        embeddings = openai_client.embeddings.create(
            input=documents,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS
        ).data
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=[e.embedding for e in embeddings]
        )

def record_until_silence():
    """Improved voice recording with better VAD"""
    print("\n🔈 Listening... (speak now)")
    
    audio_buffer = []
    voice_detected = False
    last_voice_time = time.time()
    recording_start_time = time.time()
    
    # Buffer to store recent audio chunks
    chunk_buffer = deque(maxlen=int(SAMPLE_RATE * SPEECH_BUFFER_SECONDS / CHUNK_SIZE))
    
    def callback(indata, frames, current_time, status):
        nonlocal voice_detected, last_voice_time
        
        # Store current chunk in buffer
        chunk_buffer.append(indata.copy())
        
        # Convert to 16-bit PCM for VAD
        audio_data = (indata * 32767).astype(np.int16).tobytes()
        
        # Check for voice activity
        is_speech = vad.is_speech(audio_data, SAMPLE_RATE)
        
        if is_speech:
            last_voice_time = time.time()
            if not voice_detected:
                # Voice just started - add buffered audio
                voice_detected = True
                for chunk in chunk_buffer:
                    audio_buffer.append(chunk)
                chunk_buffer.clear()
            else:
                # Ongoing voice - add current chunk
                audio_buffer.append(indata.copy())
        else:
            if voice_detected:
                # Voice was detected before - add current chunk
                audio_buffer.append(indata.copy())
    
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE,
                       blocksize=CHUNK_SIZE, dtype='float32'):
        while True:
            elapsed_silence = time.time() - last_voice_time
            if voice_detected and elapsed_silence > SILENCE_TIMEOUT:
                break
            if time.time() - recording_start_time > 10:  # Max recording time
                break
            time.sleep(0.1)
    
    if voice_detected and len(audio_buffer) > 0:
        audio_data = np.concatenate(audio_buffer)
        duration = len(audio_data) / SAMPLE_RATE
        print(f"Recorded {duration:.2f} seconds of audio")
        return audio_data, SAMPLE_RATE
    return None, SAMPLE_RATE

def transcribe_with_google(audio_data, sample_rate):
    """Use Google Speech-to-Text for wake word detection"""
    try:
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to AudioData format
        audio_data = sr.AudioData(audio_data.tobytes(), sample_rate, 2)
        
        text = google_recognizer.recognize_google(audio_data)
        print(f"Google STT detected: {text}")
        return text.strip().lower()
    except sr.UnknownValueError:
        print("Google STT could not understand audio")
        return ""
    except Exception as e:
        print(f"Google STT error: {e}")
        return ""

def transcribe_with_whisper(audio, sample_rate):
    """High-quality transcription using Whisper"""
    audio = (audio * 32767).astype(np.int16)
    temp_file = "temp_audio.wav"
    sf.write(temp_file, audio, sample_rate, subtype='PCM_16')
    
    try:
        with open(temp_file, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="en",
                response_format="text",
                temperature=0.0
            )
        return transcript.strip().lower()
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def contains_wake_word(text):
    """Enhanced wake word detection"""
    if not text:
        return False
    
    text = text.lower().strip()
    for phrase in WAKE_WORDS:
        if re.search(rf'\b{re.escape(phrase)}\b', text, re.IGNORECASE):
            return True
    return "aditya" in text

def text_to_speech(text, voice="alloy"):
    """High-quality TTS response"""
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        speed=0.95
    )
    audio_stream = io.BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)

def query_assistant(user_query):
    """Respond as Aditya Gupta with context"""
    collection = chroma_client.get_collection("knowledge_base")
    embedding = openai_client.embeddings.create(
        input=[user_query],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    
    context = "\n".join(results['documents'][0])
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": f"""You are Aditya Gupta. Respond to all questions as if you are Aditya Gupta himself, 
                using first-person perspective. Use the following context about yourself when relevant:
                {context}
                
                If the question isn't answered by the context, just say "I cannot answer that, sorry".
                """
            },
            {"role": "user", "content": user_query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    try:
        process_data_folder()
    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        return
    
    print("🌟 Aditya Gupta AI Assistant Ready (Press Ctrl+C to quit) 🌟")
    activated = False
    
    while True:
        try:
            if not activated:
                print("Listening for wake word...")
                audio, sr = record_until_silence()
                if audio is not None:
                    duration = len(audio) / sr
                    if duration >= MIN_UTTERANCE_LENGTH:
                        user_text = transcribe_with_google(audio, sr)
                        if contains_wake_word(user_text):
                            activated = True
                            text_to_speech(INTRODUCTION)
            else:
                print("Listening for query...")
                audio, sr = record_until_silence()
                if audio is not None:
                    duration = len(audio) / sr
                    if duration >= MIN_UTTERANCE_LENGTH:
                        user_text = transcribe_with_whisper(audio, sr)
                        print(f"You: {user_text}")
                        response = query_assistant(user_text)
                        print(f"Aditya: {response}")
                        text_to_speech(response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()