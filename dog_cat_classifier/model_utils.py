import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import os

class DogCatClassifier:
    def __init__(self, model_path='model/model.tflite', labels_path='model/labels.txt'):
        """
        Khởi tạo classifier hỗ trợ cả TFLite và Keras (.h5)
        Tự động xử lý kiểu dữ liệu UINT8 (Quantized) hoặc FLOAT32
        """
        self.model_path = model_path
        self.labels_path = labels_path
        
        # 1. Load Model dựa trên định dạng file
        if model_path.endswith('.tflite'):
            self.is_tflite = True
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape'][1:3] # [height, width]
            # Lấy kiểu dữ liệu đầu vào (Mục quan trọng để sửa lỗi của bạn)
            self.input_dtype = self.input_details[0]['dtype']
        else:
            self.is_tflite = False
            self.model = tf.keras.models.load_model(model_path)
            self.input_shape = (224, 224) 
            self.input_dtype = np.float32

        # 2. Load nhãn
        self.labels = self.load_labels()
        
        print(f"✅ Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Input Data Type: {self.input_dtype}")
        print(f"Labels detected: {self.labels}")

    def load_labels(self):
        """Load labels và làm sạch (loại bỏ số thứ tự nếu có)"""
        try:
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r', encoding='utf-8') as f:
                    labels = []
                    for line in f.readlines():
                        parts = line.strip().split(' ', 1)
                        label = parts[-1].lower() if len(parts) > 1 else parts[0].lower()
                        labels.append(label)
                    return labels
            return ['dog', 'cat']
        except Exception as e:
            print(f"⚠️ Error loading labels: {e}")
            return ['dog', 'cat']

    def preprocess_image(self, image):
        """
        Tiền xử lý ảnh linh hoạt theo kiểu dữ liệu model yêu cầu
        """
        # Load nếu là đường dẫn file
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Convert sang numpy array để dùng OpenCV
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Chuyển BGR sang RGB nếu ảnh từ OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image[0, 0, 0] > image[0, 0, 2]: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize theo yêu cầu của model
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

        # --- SỬA LỖI TẠI ĐÂY ---
        if self.input_dtype == np.uint8:
            # Nếu model là UINT8 (Quantized), giữ nguyên 0-255 và đổi type
            image = image.astype(np.uint8)
        else:
            # Nếu model là FLOAT32, chuẩn hóa về [-1, 1]
            image = (image.astype(np.float32) / 127.5) - 1
        
        # Thêm chiều batch (1, H, W, C)
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        """
        Dự đoán và trả về (nhãn, độ tin cậy, tất cả xác suất)
        """
        try:
            processed_image = self.preprocess_image(image)

            if self.is_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                prediction = self.model.predict(processed_image, verbose=0)[0]

            # Xử lý kết quả đầu ra
            idx = np.argmax(prediction)
            predicted_class = self.labels[idx]
            
            # Chuyển đổi xác suất nếu model là UINT8 (cần de-quantize)
            if self.input_dtype == np.uint8:
                prediction = prediction.astype(np.float32) / 255.0

            confidence = float(prediction[idx])
            class_probabilities = {label: float(prob) for label, prob in zip(self.labels, prediction)}

            return predicted_class, confidence, class_probabilities
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None, 0.0, {}

    def predict_from_file(self, file_path):
        return self.predict(file_path)

    def get_model_info(self):
        size_bytes = os.path.getsize(self.model_path)
        return {
            'input_shape': self.input_shape,
            'labels': self.labels,
            'size_mb': size_bytes / (1024 * 1024)
        }

if __name__ == "__main__":
    classifier = DogCatClassifier()