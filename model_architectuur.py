import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

def create_marker_detector_model(input_shape=(128, 128, 3), num_outputs=5):
    """
    Definieert de CNN-architectuur voor de detectie van de droppingsmarker.
    """
    inputs = Input(shape=input_shape)
    
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
    
    # De outputlaag: [x_center, y_center, width, height, confidence]
    predictions = Dense(num_outputs, activation='sigmoid', name='bounding_box_output')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    return model

