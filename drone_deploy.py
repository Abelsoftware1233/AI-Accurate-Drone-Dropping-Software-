# drone_deploy.py

import cv2
import numpy as np
import tensorflow as tf
# Je moet de DroneKit bibliotheek installeren op je drone's computer
# from dronekit import connect, VehicleMode, LocationGlobalRelative 

# --- Configuratie ---
MODEL_PATH = 'best_marker_detector.h5' # Het getrainde model
TARGET_IMAGE_SIZE = (128, 128)        # De grootte gebruikt tijdens de training
CAMERA_INDEX = 0                      # Index van de USB/CSI camera (meestal 0 of 1)
# CONNECTION_STRING = '/dev/ttyACM0'  # Voor een echte drone connectie via seriële poort

def load_ai_model():
    """Laad het getrainde AI-model van schijf."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("AI Model geladen.")
        return model
    except Exception as e:
        print(f"Fout bij het laden van model: {e}")
        return None

def predict_correction(model, frame):
    """
    Analyseert een cameraframe en voorspelt de positie van de marker.
    
    Returns: (error_x, error_y): De benodigde correctie in pixels/eenheden.
    """
    
    # 1. Beeld voorbereiden voor het model
    processed_frame = cv2.resize(frame, TARGET_IMAGE_SIZE)
    processed_frame = processed_frame.astype('float32') / 255.0
    input_tensor = np.expand_dims(processed_frame, axis=0) # Maak een batch van 1
    
    # 2. Voorspelling
    prediction = model.predict(input_tensor)[0]
    
    # 3. Ontpak de voorspelling
    x_center, y_center, width, height, confidence = prediction
    
    # --- De logica voor GPS-correctie ---
    
    if confidence > 0.7: # Alleen actie ondernemen als het model zeker is
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        
        # De voorspelde coördinaten terugschalen naar de originele framegrootte
        real_x = int(x_center * frame_width)
        real_y = int(y_center * frame_height)
        
        # Bereken de afwijking (in pixels) van het midden van het beeld
        error_x = real_x - (frame_width / 2)
        error_y = real_y - (frame_height / 2)
        
        # NOTE: Je moet deze pixel-afwijking later omzetten naar METERS of graden
        # op basis van de hoogte van de drone. Dit is de Physics/IMU-stap.
        
        return error_x, error_y, confidence
    
    return 0, 0, 0 # Geen marker gevonden of confidence te laag


def main_drone_loop():
    
    model = load_ai_model()
    if model is None:
        return

    # --- Drone Verbinding (Uitgeschakeld in dit script) ---
    # print(f"Verbinden met drone: {CONNECTION_STRING}...")
    # vehicle = connect(CONNECTION_STRING, wait_ready=True)
    # print("Drone verbonden.")

    # Camera initialiseren
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Fout: Camera kan niet geopend worden.")
        return

    print("\n--- Starten van de AI-detectielus ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kan geen frame van de camera krijgen.")
            break
        
        # Berekent de correctie
        error_x, error_y, confidence = predict_correction(model, frame)
        
        if confidence > 0.7:
            print(f"Marker gedetecteerd! Conf: {confidence:.2f}, Corr: X={error_x:.2f}, Y={error_y:.2f}")
            
            # --- Hier zou de DroneKit/MAVLink code komen ---
            # Bijvoorbeeld:
            # Als error_x positief is, moet de drone iets naar links bewegen
            # correction_in_meters_x = error_x * pixel_to_meter_factor(vehicle.location.global_relative_frame.alt)
            # vehicle.send_mavlink_command(MAV_CMD_DO_SET_POSITION, ...) 
            # -----------------------------------------------
            
        else:
            print("Geen marker of lage confidence. Gebruik GPS.")
            
        # Voorkom overbelasting van de CPU
        cv2.waitKey(1)
        
if __name__ == '__main__':
    main_drone_loop()
