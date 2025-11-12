# data_processing.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Configuratie ---
DATA_DIR = 'data/raw_images' # Map waar je ruwe marker-afbeeldingen staan
ANNOTATIONS_FILE = 'data/annotations.csv' # CSV met bounding box coördinaten
TARGET_IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.2
RANDOM_SEED = 42

def load_and_preprocess_data():
    """
    Laadt afbeeldingen en hun labels/coördinaten, en splitst ze in
    trainings- en testsets.
    """
    print("--- Starten Dataverwerking ---")
    
    # 1. Laad Annotaties
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Fout: Annotatiebestand {ANNOTATIONS_FILE} niet gevonden.")
        return None, None, None, None
        
    df = pd.read_csv(ANNOTATIONS_FILE)
    
    # 2. Laad en verwerk Afbeeldingen
    images = []
    labels = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['filename'])
        if os.path.exists(img_path):
            # Laad en resize afbeelding
            img = load_img(img_path, target_size=TARGET_IMAGE_SIZE)
            img_array = img_to_array(img)
            
            # Normalisatie (0-255 naar 0-1)
            images.append(img_array / 255.0)
            
            # De labels zijn de genormaliseerde bounding box coördinaten
            # (x_center, y_center, width, height, confidence)
            labels.append([
                row['x_center'], row['y_center'], 
                row['width'], row['height'], 
                1.0 # Stel confidence op 1 voor gelabelde data
            ])
            
    X = np.array(images)
    y = np.array(labels)

    print(f"Totaal {len(X)} afbeeldingen geladen.")
    
    # 3. Splitsen in Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"Training set: {len(X_train)} stuks. Test set: {len(X_test)} stuks.")
    print("--- Dataverwerking Voltooid ---")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Dit script wordt normaal gesproken geïmporteerd door train_model.py,
    # maar kan ook los getest worden.
    load_and_preprocess_data()
