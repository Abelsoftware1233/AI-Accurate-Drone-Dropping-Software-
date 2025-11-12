# yolo_loss.py

import tensorflow as tf
from tensorflow.keras.losses import Loss

class YoloLikeLoss(Loss):
    """
    Custom Loss Functie gespecialiseerd voor Bounding Box Regressie (zoals YOLO).
    Het bestraft de fouten in de coördinaten (x, y, w, h) sterker dan de 
    confidence score wanneer een marker daadwerkelijk aanwezig is.
    """
    def __init__(self, coord_weight=5.0, name='yolo_like_loss'):
        super().__init__(name=name)
        # Gewicht om de fouten in de bounding box coördinaten te versterken.
        self.coord_weight = coord_weight
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # y_true/y_pred shape: (batch_size, 5) -> [xc, yc, w, h, confidence]
        
        # 1. Splits de data
        # De werkelijke confidence (y_true[:, 4]) is 1.0 als er een box is, 0.0 anders (in data_processing).
        # We gebruiken dit als de 'objectness mask'.
        object_mask = y_true[:, 4]  # 1.0 als er een object is, 0.0 anders
        
        # Coördinaten (eerste 4 elementen)
        coords_true = y_true[:, :4]
        coords_pred = y_pred[:, :4]
        
        # Confidence score (laatste element)
        conf_true = y_true[:, 4]
        conf_pred = y_pred[:, 4]

        # 2. Bounding Box Coördinaten Loss (alleen waar een object is)
        # We bestraffen de coördinaatfout alleen als er een object aanwezig is (object_mask == 1).
        # De MSE wordt per element berekend, daarna gewogen.
        coord_loss_raw = self.mse(coords_true, coords_pred)
        
        # Coord Loss = Gewogen MSE * object_mask
        coord_loss = self.coord_weight * tf.reduce_sum(coord_loss_raw * tf.expand_dims(object_mask, axis=1), axis=1)

        # 3. Confidence Loss (overal)
        # Dit is een gewone MSE voor de confidence score.
        confidence_loss = self.mse(conf_true, conf_pred)
        
        # 4. Totale Loss: Som van coördinaat en confidence loss
        total_loss = coord_loss + confidence_loss
        
        # Retourneer het gemiddelde over de batch
        return tf.reduce_mean(total_loss)

