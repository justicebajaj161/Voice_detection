import os
import chromadb
from openai import OpenAI
import sounddevice as sd
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import io
import re
from dotenv import load_dotenv


load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# Constants
WAKE_WORDS = ["aditya", "hi aditya", "hey aditya", "hi aditya gupta", "hey aditya gupta"]
SAMPLE_RATE = 16000  # Optimal for Whisper
RECORD_DURATION = 5  # seconds
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
INTRODUCTION = "Hello, this is Aditya Gupta. How can I assist you today?"

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

def record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE):
    """Record audio optimized for Whisper"""
    print("\n🔈 Listening for wake word...")
    audio = sd.rec(int(duration * sample_rate),
                  samplerate=sample_rate,
                  channels=1,
                  dtype='float32')
    sd.wait()
    return audio, sample_rate

def transcribe_with_whisper(audio, sample_rate):
    """High-quality transcription using Whisper"""
    # Convert to 16-bit PCM format that Whisper prefers
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
                temperature=0.0  # Most accurate for wake words
            )
        return transcript.strip().lower()
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def contains_wake_word(text):
    """Enhanced wake word detection with Whisper"""
    if not text:
        return False
    
    text = text.lower().strip()
    
    # Check for exact matches
    for phrase in WAKE_WORDS:
        if re.search(rf'\b{re.escape(phrase)}\b', text, re.IGNORECASE):
            return True
    
    # Check for partial matches (e.g., just "aditya")
    if "aditya" in text:
        return True
    
    return False

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
            # Record and transcribe with Whisper
            audio, sr = record_audio()
            user_text = transcribe_with_whisper(audio, sr)
            
            if not user_text:
                print("No speech detected.")
                continue
                
            print(f"You: {user_text}")
            
            if not activated:
                if contains_wake_word(user_text):
                    activated = True
                    text_to_speech(INTRODUCTION)
                    continue
                print("Waiting for wake word...")
                continue
            
            # After activation, continue using Whisper for queries
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