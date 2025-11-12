# train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from model_architectuur import create_marker_detector_model
from yolo_loss import YoloLikeLoss # <<< DE NIEUWE CUSTOM LOSS WORDT GEÏMPORTEERD
import os

# --- Configuratie ---
MODEL_SAVE_PATH = 'best_marker_detector.h5'
TARGET_IMAGE_SIZE = (128, 128)
INPUT_SHAPE = TARGET_IMAGE_SIZE + (3,)
TEST_SIZE = 0.2
RANDOM_SEED = 42

def load_data(data_dir):
    """ 
    LAAD FUNCTIE VEREIST AANPASSINGEN! 
    Dit is een placeholder-functie die het resultaat van jouw echte data nabootst.
    """
    print(f"Laden van data gesimuleerd voor: {data_dir}...")
    
    # --- SIMULATIE VAN GELABELDE DATA ---
    N = 500 # Aantal afbeeldingen
    
    # X: Afbeeldingen genormaliseerd tussen 0 en 1
    X_data = np.random.rand(N, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]).astype('float32')
    
    # Y: Labels genormaliseerd tussen 0 en 1 [x_center, y_center, width, height, confidence]
    Y_data = np.random.rand(N, 5).astype('float32')
    # Zorg dat de confidence kolom (index 4) 1.0 is voor alle trainingsvoorbeelden
    Y_data[:, 4] = 1.0 
    
    print(f"Simulatie gelukt. {N} data-items geladen.")
    
    # Splitst de data in training en validatie
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_data, Y_data, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    return X_train, Y_train, X_val, Y_val

# --- Hoofd Trainingsfunctie ---
def train_detector(data_dir='./data/gelabelde_beelden', epochs=100, batch_size=32, learning_rate=1e-4):
    
    # 1. Data laden en splitsen
    X_train, Y_train, X_val, Y_val = load_data(data_dir)
    print(f"Vorm van trainingsdata: {X_train.shape}")
    print(f"Vorm van validatiedata: {X_val.shape}")
    
    # 2. Model aanmaken en compileren
    model = create_marker_detector_model(input_shape=INPUT_SHAPE)
    
    # *** BELANGRIJKE VERBETERING: GEBRUIK YoloLikeLoss IN PLAATS VAN 'mse' ***
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=YoloLikeLoss(coord_weight=5.0), # <<< DE NIEUWE LOSS FUNCTIE
        metrics=['mae', 'mse']
    )
    
    # 3. Callbacks 
    model_checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        save_best_only=True,
        monitor='val_loss', # Monitor de loss op de validatieset
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    # 4. Model trainen
    print("\n--- Starten met trainen ---")
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val), 
        callbacks=[model_checkpoint_callback, early_stopping],
        verbose=1
    )
    
    print(f"\nTraining voltooid. Beste model is opgeslagen als: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    # Pas de parameters hieronder aan naar behoefte
    train_detector(epochs=100, learning_rate=0.0001)
