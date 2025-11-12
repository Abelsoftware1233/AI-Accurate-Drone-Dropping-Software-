# evaluate_model.py

import tensorflow as tf
import numpy as np
import cv2
from data_processing import load_and_preprocess_data, TARGET_IMAGE_SIZE
from train_model import MODEL_SAVE_PATH

def evaluate_main():
    """Evalueert het getrainde model."""
    
    # 1. Laad Model
    try:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print(f"Model geladen van: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Fout bij laden model: {e}")
        return
        
    # 2. Laad Testdata
    # We gebruiken de testset uit de data_processing module
    _, X_test, _, y_test = load_and_preprocess_data()
    
    if X_test is None:
        return
        
    print("--- Starten Evaluatie ---")
    
    # 3. Evalueer de Loss en Metrics
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\n--- Resultaten op Testset ---")
    print(f"Verlies (Loss - MSE): {loss:.4f}")
    print(f"Gemiddelde Absolute Fout (MAE): {mae:.4f}")
    
    # 4. Visuele Test (voor een beter inzicht)
    print("\nVisuele inspectie van de eerste 5 testbeelden:")
    
    for i in range(5):
        # Voorspelling
        input_tensor = np.expand_dims(X_test[i], axis=0)
        prediction = model.predict(input_tensor, verbose=0)[0]
        
        # Terugschalen naar 0-255 en CV2-formaat
        image = (X_test[i] * 255).astype(np.uint8)
        
        # Bounding box coördinaten terug naar pixels
        w, h = TARGET_IMAGE_SIZE
        
        # Voorspelde Bounding Box (Rood)
        pred_x, pred_y, pred_w, pred_h, _ = prediction
        x1_p = int((pred_x - pred_w / 2) * w)
        y1_p = int((pred_y - pred_h / 2) * h)
        x2_p = int((pred_x + pred_w / 2) * w)
        y2_p = int((pred_y + pred_h / 2) * h)
        cv2.rectangle(image, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 2) # Rood
        
        # Werkelijke Bounding Box (Groen)
        true_x, true_y, true_w, true_h, _ = y_test[i]
        x1_t = int((true_x - true_w / 2) * w)
        y1_t = int((true_y - true_h / 2) * h)
        x2_t = int((true_x + true_w / 2) * w)
        y2_t = int((true_y + true_h / 2) * h)
        cv2.rectangle(image, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 1) # Groen (dunnere lijn)
        
        cv2.imshow(f"Test Image {i+1} (Rood=Voorspeld, Groen=Werkelijk)", image)
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    evaluate_main()
