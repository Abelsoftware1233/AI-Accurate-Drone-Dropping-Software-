import cv2
import numpy as np
import tensorflow as tf
# Importeer de custom loss, anders kan TF het model niet laden!
from yolo_loss import YoloLikeLoss 
# from dronekit import connect, VehicleMode, LocationGlobalRelative # Deactiveer DroneKit voor desktop testen

# --- Configuratie ---
MODEL_PATH = 'best_marker_detector.h5' 
TARGET_IMAGE_SIZE = (128, 128)        
CAMERA_INDEX = 0                      
# CONNECTION_STRING = '/dev/ttyACM0'  

def load_ai_model():
    """Laad het getrainde AI-model van schijf."""
    try:
        # Laad het model met de custom loss functie
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={'YoloLikeLoss': YoloLikeLoss}
        )
        print("AI Model geladen.")
        return model
    except Exception as e:
        print(f"Fout bij het laden van model: {e}")
        return None

def predict_correction(model, frame):
    """
    Analyseert een cameraframe en voorspelt de positie van de marker.
    """
    
    # 1. Beeld voorbereiden
    processed_frame = cv2.resize(frame, TARGET_IMAGE_SIZE)
    processed_frame = processed_frame.astype('float32') / 255.0
    input_tensor = np.expand_dims(processed_frame, axis=0)
    
    # 2. Voorspelling
    prediction = model.predict(input_tensor, verbose=0)[0]
    
    # 3. Ontpak de voorspelling
    x_center, y_center, width, height, confidence = prediction
    
    if confidence > 0.7:
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        
        # De voorspelde coördinaten terugschalen
        real_x = int(x_center * frame_width)
        real_y = int(y_center * frame_height)
        
        # Bereken de afwijking (in pixels) van het midden
        error_x = real_x - (frame_width / 2)
        error_y = real_y - (frame_height / 2)
        
        # VISUALISATIE (voor debuggen)
        cv2.circle(frame, (real_x, real_y), 10, (0, 255, 255), 2)
        cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 0, 0), 1)

        return error_x, error_y, confidence, frame
    
    return 0, 0, 0, frame # Geen marker gevonden


def main_drone_loop():
    
    model = load_ai_model()
    if model is None:
        return

    # Camera initialiseren
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Fout: Camera kan niet geopend worden.")
        return

    print("\n--- Starten van de AI-detectielus (Druk op 'q' om te stoppen) ---")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kan geen frame van de camera krijgen.")
            break
        
        # Berekent de correctie en voegt debug-info toe aan frame
        error_x, error_y, confidence, frame_processed = predict_correction(model, frame)
        
        if confidence > 0.7:
            print(f"Marker gedetecteerd! Conf: {confidence:.2f}, Corr: X={error_x:.2f}, Y={error_y:.2f}")
            
            # TODO: IMPLEMENTEER HIER DE PIXEL-NAAR-METER/PID-LOGICA
            # vehicle.send_mavlink_command(...)
            
        else:
            cv2.putText(frame_processed, "Zoeken...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Toon het beeld (deactiveer dit voor headless deployment)
        cv2.imshow("Drone Feed", frame_processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main_drone_loop()
