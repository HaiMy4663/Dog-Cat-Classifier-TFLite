import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from model_utils import DogCatClassifier
import threading

class DogCatClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog vs Cat Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Khởi tạo classifier
        self.classifier = DogCatClassifier()
        
        # Biến lưu ảnh hiện tại
        self.current_image = None
        self.current_image_path = None
        self.photo_image = None
        
        # Tạo giao diện
        self.setup_ui()
        
    def setup_ui(self):
        """Tạo giao diện ứng dụng"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#4CAF50", height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🐶 Dog vs Cat Classifier 🐱",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#4CAF50"
        )
        title_label.pack(pady=20)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Left panel - Image display
        left_frame = tk.Frame(main_frame, bg="white", relief="solid", bd=1)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Image display
        self.image_label = tk.Label(
            left_frame,
            text="No Image Selected",
            font=("Arial", 12),
            bg="white",
            fg="#666",
            width=40,
            height=20
        )
        self.image_label.pack(padx=10, pady=10)
        
        # Right panel - Controls and results
        right_frame = tk.Frame(main_frame, bg="white", relief="solid", bd=1)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Control buttons
        btn_frame = tk.Frame(right_frame, bg="white")
        btn_frame.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(
            btn_frame,
            text="📁 Upload Image",
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            command=self.upload_image
        )
        upload_btn.pack(side="left", padx=5)
        
        # Camera button
        camera_btn = tk.Button(
            btn_frame,
            text="📷 Use Camera",
            font=("Arial", 12),
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=10,
            command=self.use_camera
        )
        camera_btn.pack(side="left", padx=5)
        
        # Classify button
        classify_btn = tk.Button(
            btn_frame,
            text="🔍 Classify",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            command=self.classify_image
        )
        classify_btn.pack(side="left", padx=5)
        
        # Results frame
        results_frame = tk.LabelFrame(
            right_frame,
            text="Classification Results",
            font=("Arial", 14, "bold"),
            bg="white",
            fg="#333"
        )
        results_frame.pack(fill="both", expand=True, padx=10, pady=20)
        
        # Predicted class
        self.result_label = tk.Label(
            results_frame,
            text="Waiting for image...",
            font=("Arial", 18, "bold"),
            bg="white",
            fg="#666"
        )
        self.result_label.pack(pady=20)
        
        # Confidence
        self.confidence_label = tk.Label(
            results_frame,
            text="Confidence: 0%",
            font=("Arial", 14),
            bg="white",
            fg="#666"
        )
        self.confidence_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            results_frame,
            length=300,
            mode='determinate'
        )
        self.progress.pack(pady=20)
        
        # Probabilities frame
        self.probs_frame = tk.Frame(results_frame, bg="white")
        self.probs_frame.pack(pady=10)
        
        # Model info
        info_frame = tk.LabelFrame(
            right_frame,
            text="Model Information",
            font=("Arial", 12),
            bg="white",
            fg="#666"
        )
        info_frame.pack(fill="x", padx=10, pady=10)
        
        model_info = self.classifier.get_model_info()
        info_text = f"Input: {model_info['input_shape'][0]}x{model_info['input_shape'][1]}\n"
        info_text += f"Labels: {', '.join(model_info['labels'])}"
        
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=("Arial", 10),
            bg="white",
            justify="left"
        )
        info_label.pack(pady=5)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)
    
    def upload_image(self):
        """Mở dialog để chọn ảnh"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load và hiển thị ảnh"""
        try:
            self.current_image_path = file_path
            
            # Mở ảnh với PIL
            image = Image.open(file_path)
            
            # Resize để hiển thị
            display_size = (400, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert sang PhotoImage cho tkinter
            self.photo_image = ImageTk.PhotoImage(image)
            
            # Hiển thị ảnh
            self.image_label.config(
                image=self.photo_image,
                text=""
            )
            
            # Update status
            filename = os.path.basename(file_path)
            self.result_label.config(
                text=f"Loaded: {filename}",
                fg="#2196F3"
            )
            self.confidence_label.config(text="Click 'Classify' to analyze")
            
            # Reset progress
            self.progress['value'] = 0
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")
    
    def use_camera(self):
        """Sử dụng webcam (chức năng nâng cao)"""
        messagebox.showinfo("Webcam", "Webcam feature will be implemented soon!")
        # Gợi ý: Sử dụng OpenCV để capture từ webcam
    
    def classify_image(self):
        """Phân loại ảnh hiện tại"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        # Hiển thị loading
        self.result_label.config(text="Analyzing...", fg="#FF9800")
        self.root.update()
        
        # Chạy prediction trong thread riêng để không block UI
        def predict_thread():
            try:
                predicted_class, confidence, probs = self.classifier.predict_from_file(
                    self.current_image_path
                )
                
                # Update UI trong main thread
                self.root.after(0, self.update_results, predicted_class, confidence, probs)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        # Start prediction thread
        thread = threading.Thread(target=predict_thread)
        thread.daemon = True
        thread.start()
    
    def update_results(self, predicted_class, confidence, probs):
        """Update kết quả lên UI"""
        # Update result labels
        self.result_label.config(
            text=f"Prediction: {predicted_class.upper()}",
            fg="#4CAF50" if predicted_class == "dog" else "#FF5722"
        )
        
        self.confidence_label.config(
            text=f"Confidence: {confidence:.2%}",
            fg="#333"
        )
        
        # Update progress bar
        self.progress['value'] = confidence * 100
        
        # Clear old probabilities
        for widget in self.probs_frame.winfo_children():
            widget.destroy()
        
        # Hiển thị probabilities
        tk.Label(
            self.probs_frame,
            text="Probabilities:",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(anchor="w")
        
        # Tạo bar chart cho probabilities
        for label, prob in probs.items():
            frame = tk.Frame(self.probs_frame, bg="white")
            frame.pack(fill="x", pady=2)
            
            # Label
            tk.Label(
                frame,
                text=label.capitalize(),
                width=10,
                anchor="w",
                bg="white"
            ).pack(side="left")
            
            # Progress bar
            prob_bar = ttk.Progressbar(
                frame,
                length=200,
                mode='determinate'
            )
            prob_bar.pack(side="left", padx=5)
            prob_bar['value'] = prob * 100
            
            # Percentage
            tk.Label(
                frame,
                text=f"{prob:.2%}",
                width=10,
                bg="white"
            ).pack(side="left")

def main():
    root = tk.Tk()
    app = DogCatClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
