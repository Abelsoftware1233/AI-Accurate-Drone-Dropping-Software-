import tensorflow as tf
import numpy as np
import cv2
from data_processing import load_and_preprocess_data, TARGET_IMAGE_SIZE
from yolo_loss import YoloLikeLoss # Nodig om het model correct te laden
import os

MODEL_PATH = 'best_marker_detector.h5'

def evaluate_main():
    """Evalueert het getrainde model op de testdata."""
    
    # 1. Laad Model (met Custom Object)
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={'YoloLikeLoss': YoloLikeLoss}
        )
        print(f"Model geladen van: {MODEL_PATH}")
    except Exception as e:
        print(f"Fout bij laden model: {e}")
        return
        
    # 2. Laad Validatiedata (gebruikt als testset)
    # We gebruiken de validatieset X_val, Y_val om te evalueren
    _, X_val, _, Y_val = load_and_preprocess_data()
    
    if X_val is None:
        return
        
    print("--- Starten Evaluatie ---")
    
    # 3. Evalueer de Loss en Metrics
    loss, mae, mse = model.evaluate(X_val, Y_val, verbose=1)
    
    print(f"\n--- Resultaten op Validatieset ---")
    print(f"Loss: {loss:.4f}, MAE: {mae:.4f}")
    
    # 4. Visuele Test (Rood=Voorspeld, Groen=Werkelijk)
    print("\nVisuele inspectie van de eerste 5 beelden (sluit venster om door te gaan):")
    
    for i in range(5):
        input_tensor = np.expand_dims(X_val[i], axis=0)
        prediction = model.predict(input_tensor, verbose=0)[0]
        
        image = (X_val[i] * 255).astype(np.uint8)
        w, h = TARGET_IMAGE_SIZE
        
        # Voorspelde Bounding Box (Rood)
        pred_x, pred_y, pred_w, pred_h, _ = prediction
        x1_p = int((pred_x - pred_w / 2) * w)
        y1_p = int((pred_y - pred_h / 2) * h)
        x2_p = int((pred_x + pred_w / 2) * w)
        y2_p = int((pred_y + pred_h / 2) * h)
        cv2.rectangle(image, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 2) 
        
        # Werkelijke Bounding Box (Groen)
        true_x, true_y, true_w, true_h, _ = Y_val[i]
        x1_t = int((true_x - true_w / 2) * w)
        y1_t = int((true_y - true_h / 2) * h)
        x2_t = int((true_x + true_w / 2) * w)
        y2_t = int((true_y + true_h / 2) * h)
        cv2.rectangle(image, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 1) 
        
        cv2.imshow(f"Test Image {i+1}", image)
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    evaluate_main()
