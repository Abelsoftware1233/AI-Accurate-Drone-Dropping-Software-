import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Configuratie ---
DATA_DIR = 'data/raw_images' 
ANNOTATIONS_FILE = 'data/annotations.csv' 
TARGET_IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.2
RANDOM_SEED = 42

def load_and_preprocess_data():
    """
    Laadt afbeeldingen en hun labels/coördinaten, en splitst ze in
    trainings- en testsets.
    """
    print("--- Starten Dataverwerking ---")
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Fout: Annotatiebestand {ANNOTATIONS_FILE} niet gevonden.")
        return None, None, None, None
        
    df = pd.read_csv(ANNOTATIONS_FILE)
    
    images = []
    labels = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['filename'])
        if os.path.exists(img_path):
            # Laad en resize de afbeelding
            img = load_img(img_path, target_size=TARGET_IMAGE_SIZE)
            img_array = img_to_array(img)
            
            # Normalisatie van de pixelwaarden (0-255 naar 0-1)
            images.append(img_array / 255.0)
            
            # De labels zijn de genormaliseerde coördinaten + confidence (1.0 in training)
            labels.append([
                row['x_center'], row['y_center'], 
                row['width'], row['height'], 
                1.0 # Object is aanwezig
            ])
            
    X_data = np.array(images)
    Y_data = np.array(labels)

    print(f"Totaal {len(X_data)} afbeeldingen geladen.")
    
    # Splitsen in Train/Test
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_data, Y_data, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"Training set: {len(X_train)} stuks. Validatie set: {len(X_val)} stuks.")
    print("--- Dataverwerking Voltooid ---")
    return X_train, X_val, Y_train, Y_val

if __name__ == '__main__':
    load_and_preprocess_data()

