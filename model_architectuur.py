# model_architectuur.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, BatchNormalization

def create_marker_detector_model(input_shape=(128, 128, 3), num_outputs=5):
    """
    Definieert de CNN-architectuur voor de detectie van de droppingsmarker.
    
    Args:
        input_shape (tuple): De afmetingen van de inputafbeeldingen (hoogte, breedte, kanalen).
        num_outputs (int): Het aantal voorspelde waarden per box: 
                           [x_center, y_center, width, height, confidence].
                               
    Returns:
        tf.keras.Model: Het te compileren Keras-model.
    """
    
    inputs = Input(shape=input_shape)
    
    # Gebruik van BatchNormalization verbetert de training
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)
    
    # De output laag: 5 waarden voor de bounding box en confidence
    predictions = Dense(num_outputs, activation='sigmoid', name='bounding_box_output')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    return model

if __name__ == '__main__':
    # Voorbeeldgebruik en samenvatting van het model
    model = create_marker_detector_model()
    model.summary()
    print("\nModel architectuur is gedefinieerd in 'model_architectuur.py'.")
  
