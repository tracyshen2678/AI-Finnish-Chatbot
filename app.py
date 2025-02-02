import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import sys
sys.stderr = open(os.devnull, 'w')
import warnings
import torch
from contextlib import redirect_stdout, redirect_stderr
import io
import pygame
from gtts import gTTS
from transformers import pipeline, logging, AutoTokenizer, AutoModelForCausalLM

# Suppress all warnings and Pygame welcome message
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Suppress pygame initialization messages
pygame_output = io.StringIO()
with redirect_stdout(pygame_output), redirect_stderr(pygame_output):
    pygame.init()
    pygame.mixer.init()

class FinishChatbot:
    def __init__(self):
        # Force CPU usage
        self.device = torch.device("cpu")

        # Initialize speech recognition
        self.speech_recognizer = pipeline(
            "automatic-speech-recognition",
            model="Finnish-NLP/whisper-large-finnish-v3",
            device=self.device
        )

        # Initialize Finnish GPT model
        model_name = "Finnish-NLP/gpt2-medium-finnish"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def transcribe_audio(self, audio_file):
        """Convert speech to text"""
        try:
            transcription = self.speech_recognizer(audio_file)
            return transcription["text"]
        except Exception as e:
            print(f"Error transcribing: {e}")
            return None

    def generate_response(self, input_text):
        """Generate text response using Finnish GPT"""
        try:
            # Prepare input
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.device)

            # Generate response
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=100,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Process response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.replace(input_text, "").strip()

            # Stop at first punctuation mark
            for punct in [".", "!", "?"]:
                if punct in response:
                    response = response.split(punct)[0] + punct
                    break

            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def text_to_speech(self, text):
        """Convert text to speech"""
        try:
            # Generate audio file
            tts = gTTS(text, lang='fi')
            tts.save("response.mp3")

            # Play audio
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Clean up
            os.remove("response.mp3")
            return True
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            return False

    def chat_loop(self):
        """Main chat loop"""
        print("Finnish Chatbot launched! Please select a mode or type 'quit' to exit")

        while True:
            # Get input method
            mode = input("Please choose input type (1:Audio 2:Text): ").strip()

            if mode == "1":
                # Voice input
                audio_file = input("Please enter audio file path: ").strip()
                if not os.path.exists(audio_file):
                    print("File does not exist!")
                    continue

                print("Transcribing...")
                input_text = self.transcribe_audio(audio_file)
                if not input_text:
                    continue
                print(f"Recognized text: {input_text}")

            elif mode == "2":
                # Text input
                input_text = input("You: ").strip()
                if input_text.lower() == 'quit':
                    break
            else:
                print("Invalid selection!")
                continue

            # Generate and speak response
            print("Thinking...")
            response = self.generate_response(input_text)
            if response:
                print(f"AI: {response}")
                print("Generating speech...")
                self.text_to_speech(response)

if __name__ == "__main__":
    # Suppress initial pygame messages
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        chatbot = FinishChatbot()
    chatbot.chat_loop()

