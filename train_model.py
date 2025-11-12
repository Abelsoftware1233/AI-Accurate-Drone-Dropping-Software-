# train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_architectuur import create_marker_detector_model
import os

# --- Plaats jouw data-laadlogica hier ---
def load_data(data_dir):
    """ LAAD FUNCTIE VEREIST AANPASSINGEN! 
        Laadt de afbeeldingen (X) en de gelabelde bounding box coördinaten (Y).
        De Y-waarden moeten genormaliseerd zijn (bijv. tussen 0 en 1).
    """
    print(f"Laden van data uit: {data_dir}...")
    # Dummy data voor demonstratie:
    X_train = np.random.rand(100, 128, 128, 3).astype('float32') # 100 afbeeldingen
    Y_train = np.random.rand(100, 5).astype('float32')          # 100 labels: [x, y, w, h, conf]
    
    # Normaliseer de inputafbeeldingen
    X_train /= 255.0
    
    return X_train, Y_train

# --- Hoofd Trainingsfunctie ---
def train_detector(data_dir='path/to/your/data', epochs=50, batch_size=32):
    
    # 1. Data laden
    X_train, Y_train = load_data(data_dir)
    print(f"Vorm van trainingsdata: {X_train.shape}")
    
    # 2. Model aanmaken en compileren
    input_shape = X_train.shape[1:]
    model = create_marker_detector_model(input_shape=input_shape)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse', # Mean Squared Error is standaard voor regressie (bounding box coördinaten)
        metrics=['mae'] # Mean Absolute Error
    )
    
    # 3. Callbacks (Om de beste versie op te slaan)
    checkpoint_filepath = 'best_marker_detector.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='loss',
        mode='min'
    )
    
    # 4. Model trainen
    print("\n--- Starten met trainen ---")
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[model_checkpoint_callback, EarlyStopping(patience=10)],
        verbose=1 # Toont de voortgang
    )
    
    print(f"\nTraining voltooid. Beste model is opgeslagen als: {checkpoint_filepath}")

if __name__ == '__main__':
    # Zorg dat de data-directory bestaat en de gelabelde data bevat
    # Pas de paden en parameters hieronder aan!
    train_detector(data_dir='./data/gelabelde_beelden', epochs=100)
  
