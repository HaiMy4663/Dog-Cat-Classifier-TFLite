import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, auc, f1_score
)
from sklearn.calibration import calibration_curve
from model_utils import DogCatClassifier

# --- MỤC 1: ACCURACY ---
def calculate_accuracy(classifier, test_dir):
    y_true, y_pred = [], []
    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                pred_class, _, _ = classifier.predict_from_file(os.path.join(class_dir, filename))
                # KIỂM TRA DỮ LIỆU HỢP LỆ
                if pred_class is not None:
                    y_true.append(class_name)
                    y_pred.append(pred_class)
    
    if len(y_true) == 0:
        print("❌ Không có dữ liệu dự đoán hợp lệ để tính Accuracy.")
        return 0
        
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return acc

# --- MỤC 2: PRECISION, RECALL, F1 ---
def calculate_precision_recall_f1(classifier, test_dir):
    y_true, y_pred = [], []
    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                pred_class, _, _ = classifier.predict_from_file(os.path.join(class_dir, filename))
                if pred_class is not None:
                    y_true.append(class_name)
                    y_pred.append(pred_class)
    
    if len(y_true) == 0: return None
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=['dog', 'cat'], average=None, zero_division=0)
    for i, name in enumerate(['dog', 'cat']):
        print(f"Class {name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    return precision, recall, f1

# --- MỤC 3: CONFUSION MATRIX ---
def plot_confusion_matrix(classifier, test_dir, save_path=None):
    y_true, y_pred = [], []
    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                pred_class, _, _ = classifier.predict_from_file(os.path.join(class_dir, filename))
                if pred_class is not None:
                    y_true.append(class_name)
                    y_pred.append(pred_class)
    
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=['dog', 'cat'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dog', 'cat'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        if save_path: plt.savefig(save_path)
        plt.show()

# --- MỤC 4: ROC AUC ---
def plot_roc_auc(classifier, test_dir):
    y_true, y_scores = [], []
    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                res = classifier.predict_from_file(os.path.join(class_dir, filename))
                if res[0] is not None:
                    probs = res[2]
                    y_scores.append(probs.get('dog', 0.0))
                    y_true.append(1 if class_name == 'dog' else 0)
    
    if len(y_true) > 0:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

# --- MỤC 5: INFERENCE TIME ---
def measure_inference_time(classifier, image_path, num_runs=10):
    times = []
    # Chạy nháp 1 lần để khởi động model
    classifier.predict_from_file(image_path)
    
    for _ in range(num_runs):
        start = time.perf_counter()
        classifier.predict_from_file(image_path)
        times.append(time.perf_counter() - start)
    avg_time = sum(times) / num_runs
    print(f"Average inference time: {avg_time*1000:.2f} ms")

# --- MỤC 6: MODEL SIZE ---
def get_model_size(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

# --- MỤC 7: CALIBRATION ---
def plot_calibration_curve(classifier, test_dir):
    y_true, y_scores = [], []
    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                pred_class, conf, _ = classifier.predict_from_file(os.path.join(class_dir, filename))
                if pred_class is not None:
                    y_scores.append(conf)
                    # 1 nếu dự đoán đúng lớp thật, 0 nếu sai
                    y_true.append(1 if pred_class == class_name else 0)
    
    if len(y_true) > 0:
        f_pos, m_pred = calibration_curve(y_true, y_scores, n_bins=5)
        plt.figure()
        plt.plot(m_pred, f_pos, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.title("Calibration Curve")
        plt.legend()
        plt.show()

# --- MỤC 8: THRESHOLD TUNING ---
def find_optimal_threshold(classifier, test_dir):
    y_true, y_scores_dog = [], []
    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                res = classifier.predict_from_file(os.path.join(class_dir, filename))
                if res[0] is not None:
                    probs = res[2]
                    y_scores_dog.append(probs.get('dog', 0.0))
                    y_true.append(1 if class_name == 'dog' else 0)
    
    if len(y_true) > 0:
        thresholds = np.linspace(0.01, 0.99, 100)
        best_f1, best_thresh = 0, 0.5
        y_true_np = np.array(y_true)
        y_scores_np = np.array(y_scores_dog)
        
        for thresh in thresholds:
            f1 = f1_score(y_true_np, (y_scores_np >= thresh).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        print(f"Optimal threshold: {best_thresh:.3f} with F1 = {best_f1:.4f}")

# --- MỤC 9: SCRIPT HOÀN CHỈNH ---
def main():
    parser = argparse.ArgumentParser(description='Evaluate Dog/Cat Classifier')
    parser.add_argument('--test_dir', required=True, help='Path to test directory')
    parser.add_argument('--model', default='model/model.tflite', help='Path to model file')
    parser.add_argument('--labels', default='model/labels.txt', help='Path to labels file')
    args = parser.parse_args()

    # Khởi tạo classifier
    classifier = DogCatClassifier(model_path=args.model, labels_path=args.labels)

    print("\n=== Accuracy ===")
    calculate_accuracy(classifier, args.test_dir)

    print("\n=== Precision/Recall/F1 ===")
    calculate_precision_recall_f1(classifier, args.test_dir)

    print("\n=== Confusion Matrix ===")
    plot_confusion_matrix(classifier, args.test_dir, save_path='confusion.png')

    print("\n=== ROC AUC ===")
    plot_roc_auc(classifier, args.test_dir)

    print("\n=== Inference Time ===")
    try:
        # Tự động tìm 1 ảnh trong folder dog để test thời gian
        dog_folder = os.path.join(args.test_dir, 'dog')
        if os.path.exists(dog_folder) and len(os.listdir(dog_folder)) > 0:
            sample_img = os.path.join(dog_folder, os.listdir(dog_folder)[0])
            measure_inference_time(classifier, sample_img)
        else:
            print("Không tìm thấy ảnh mẫu trong folder 'dog' để đo thời gian.")
    except Exception as e:
        print(f"Lỗi khi đo thời gian: {e}")

    print("\n=== Model Size ===")
    get_model_size(args.model)

    print("\n=== Calibration ===")
    plot_calibration_curve(classifier, args.test_dir)

    print("\n=== Threshold Tuning ===")
    find_optimal_threshold(classifier, args.test_dir)

if __name__ == "__main__":
    main()