import sys
import os
import argparse

# Menambahkan direktori src ke PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config import Config  # Import langsung karena berada di folder yang sama
from model import UniversityChatbot

def main():
    parser = argparse.ArgumentParser(description='Train chatbot model')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                        help='Specific checkpoint to resume from (e.g., "best_model_epoch_3")')
    args = parser.parse_args()
    
    print("Initializing chatbot...")
    chatbot = UniversityChatbot()
    
    if args.resume:
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}...")
            chatbot.load_model(model_path=os.path.join(Config.MODEL_PATH, args.checkpoint))
        else:
            print("Loading latest checkpoint...")
            chatbot.load_model(model_type='latest')
        print("Resuming training...")
    else:
        print("Starting new training...")
    
    try:
        chatbot.train(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, 
                        resume=args.resume)
        
        print("Saving model...")
        chatbot.save_model(is_final=True)
        
        print("Testing model...")
        test_questions = [
            "tampilkan seluruh informasi mengenai Rayhan Pratama",
            "berapa NPM Rayhan Pratama?",
            "apa jurusan Rayhan Pratama?"
        ]
        
        for question in test_questions:
            print(f"\nQ: {question}")
            answer = chatbot.answer_question(question)
            print(f"A: {answer}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        # Save current model state
        chatbot.save_model(is_final=False, metadata={'interrupted': True})
        sys.exit(0)

if __name__ == "__main__":
    main()