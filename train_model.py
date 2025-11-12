import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_architectuur import create_marker_detector_model
from data_processing import load_and_preprocess_data, TARGET_IMAGE_SIZE
from yolo_loss import YoloLikeLoss 

# --- Configuratie ---
MODEL_SAVE_PATH = 'best_marker_detector.h5'
INPUT_SHAPE = TARGET_IMAGE_SIZE + (3,)
LEARNING_RATE = 1e-4
EPOCHS = 100

def train_detector(epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=32):
    
    # 1. Data laden en splitsen
    X_train, X_val, Y_train, Y_val = load_and_preprocess_data()
    
    if X_train is None:
        return
        
    # 2. Model aanmaken en compileren
    model = create_marker_detector_model(input_shape=INPUT_SHAPE)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=YoloLikeLoss(coord_weight=5.0), # De gespecialiseerde Loss Functie
        metrics=['mae', 'mse']
    )
    
    # 3. Callbacks voor opslaan en stoppen
    model_checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        save_best_only=True,
        monitor='val_loss', 
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
    
    print(f"\nTraining voltooid. Beste model opgeslagen als: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_detector()
