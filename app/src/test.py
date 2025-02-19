
# API/app/src/test.py
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model import UniversityChatbot

def test_model(model_type='final'):
    """
    Fungsi testing dengan opsi pemilihan model
    
    Args:
        model_type (str): Tipe model yang akan digunakan
                            'final' - gunakan final_model
                            'latest' - gunakan checkpoint terbaru
                            'best' - gunakan model dengan performa terbaik
    """
    print("Loading model...")
    chatbot = UniversityChatbot()
    chatbot.load_model(model_type=model_type)
    
    while True:
        question = input("\nMasukkan pertanyaan (atau 'quit' untuk keluar): ")
        if question.lower() == 'quit':
            break
            
        answer = chatbot.answer_question(question)
        print(f"Jawaban: {answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test chatbot model')
    parser.add_argument('--model-type', type=str, default='final',
                        choices=['final', 'latest', 'best'],
                        help='Tipe model yang akan digunakan')
    
    args = parser.parse_args()
    test_model(model_type=args.model_type)