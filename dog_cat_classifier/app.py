import sys
import os
import argparse
from model_utils import DogCatClassifier

def main():
    parser = argparse.ArgumentParser(description="Dog vs Cat Classifier")
    parser.add_argument("--mode", choices=["gui", "web", "cli"], default="gui",
                       help="Application mode: gui (Tkinter), web (Streamlit), cli (Command line)")
    parser.add_argument("--image", type=str, help="Path to image for CLI mode")
    parser.add_argument("--model", type=str, default="model/model.tflite", 
                       help="Path to model file")
    parser.add_argument("--labels", type=str, default="model/labels.txt",
                       help="Path to labels file")
    
    args = parser.parse_args()
    
    # CLI Mode
    if args.mode == "cli":
        if not args.image:
            print("Error: Please provide an image path with --image")
            sys.exit(1)
        
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            sys.exit(1)
        
        # Initialize classifier
        classifier = DogCatClassifier(args.model, args.labels)
        
        # Predict
        predicted_class, confidence, probs = classifier.predict_from_file(args.image)
        
        print("\n" + "="*50)
        print("DOG vs CAT CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Image: {args.image}")
        print(f"Prediction: {predicted_class.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print("\nProbabilities:")
        for class_name, prob in probs.items():
            bar = "█" * int(prob * 20)
            print(f"  {class_name:10s} {prob:6.2%} {bar}")
        print("="*50)
    
    # GUI Mode
    elif args.mode == "gui":
        try:
            from gui import main as gui_main
            gui_main()
        except ImportError as e:
            print(f"GUI mode requires tkinter. Error: {e}")
            print("Try running in web mode or install tkinter")
    
    # Web Mode
    elif args.mode == "web":
        print("Starting web application...")
        print("Open your browser and go to http://localhost:8501")
        os.system("streamlit run web_app.py")

if __name__ == "__main__":
    main()
