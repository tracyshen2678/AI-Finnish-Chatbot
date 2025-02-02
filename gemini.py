import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import sys
import warnings
import torch
import io
import pygame
import requests
from gtts import gTTS
from transformers import pipeline, logging


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Initialize Pygame (for playing AI voice)
pygame.init()
pygame.mixer.init()

# 1Gemini API Key
GEMINI_API_KEY = "AIzaSyB8pntNzd6CHYGYGmD9ve2kY7YppX6U6Lo"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"


class FinnishChatbot:
    def __init__(self):
        """ Initialize chatbot, load speech recognition and synthesis modules """
        self.device = torch.device("cpu")  # Force CPU usage

        # Speech to text (Whisper Finnish large model)
        self.speech_recognizer = pipeline(
            "automatic-speech-recognition",
            model="Finnish-NLP/whisper-large-finnish-v3",
            device=-1
        )

    def transcribe_audio(self, audio_file):
        """ 🎤 Speech to text """
        try:
            print("🎙️ Recognizing speech...")
            transcription = self.speech_recognizer(audio_file)
            text = transcription["text"]
            print(f"📝 Recognition result: {text}")
            return text
        except Exception as e:
            print(f"❌ Speech recognition failed: {e}")
            return None

    def generate_response(self, input_text):
        """ 🤖 Generate AI response via Gemini API """
        try:
            payload = {
                "contents": [
                    {"parts": [{"text": input_text}]}
                ]
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(GEMINI_URL, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                ai_response = data["candidates"][0]["content"]["parts"][0]["text"]
                return ai_response.strip()
            else:
                print(f"❌ API request failed: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"❌ Failed to generate AI response: {e}")
            return None

    def text_to_speech(self, text):
        """ 🔊 Convert AI response to speech """
        try:
            tts = gTTS(text, lang='fi')
            tts.save("response.mp3")

            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            os.remove("response.mp3")  # Clean up temporary file
            return True
        except Exception as e:
            print(f"❌ Speech synthesis failed: {e}")
            return False

    def chat_loop(self):
        """ 🗣️ Interactive mode, supports voice and text input """
        print("🚀 Finnish AI Chatbot started!\n Choose input method: 1️⃣ Voice input 2️⃣ Text input (type 'quit' to exit)")

        while True:
            mode = input("🔹 Select input mode (1: Voice 🎙️ / 2: Text 📝): ").strip()

            if mode == "1":
                audio_file = input("📂 Enter audio file path: ").strip()
                if not os.path.exists(audio_file):
                    print("❌ File doesn't exist, please check the path!")
                    continue

                input_text = self.transcribe_audio(audio_file)
                if not input_text:
                    continue
                print(f"👤 You said: {input_text}")

            elif mode == "2":
                input_text = input("👤 You: ").strip()
                if input_text.lower() == "quit":
                    break
            else:
                print("⚠️ Invalid choice, please enter 1 or 2!")
                continue

            print("🤖 AI thinking...")
            response = self.generate_response(input_text)
            if response:
                print(f"🤖 AI: {response}")
                print("🔊 Playing AI voice...")
                self.text_to_speech(response)
            else:
                print("⚠️ AI response failed!")


if __name__ == "__main__":
    chatbot = FinnishChatbot()
    chatbot.chat_loop()
