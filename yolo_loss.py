# yolo_loss.py - VERBETERDE VERSIE VOOR ROBUUSTHEID

import tensorflow as tf
from tensorflow.keras.losses import Loss

class YoloLikeLoss(Loss):
    """
    Custom Loss Functie voor Bounding Box Regressie (zoals YOLO).
    Bestraft coördinatenfouten (x, y, w, h) sterker en weegt No-Object Loss lichter.
    """
    def __init__(self, coord_weight=5.0, no_object_weight=0.5, name='yolo_like_loss'):
        super().__init__(name=name)
        # Gewicht voor de coördinatenfout (moet hoog zijn)
        self.coord_weight = coord_weight
        # Gewicht voor het geval er GEEN object is (moet laag zijn)
        self.no_object_weight = no_object_weight
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # y_true/y_pred shape: (batch_size, 5) -> [xc, yc, w, h, confidence]
        
        # 1. Bepaal de maskers
        object_mask = y_true[:, 4]        # 1.0 als object aanwezig (Positieve voorbeelden)
        no_object_mask = 1.0 - object_mask # 1.0 als object afwezig (Negatieve voorbeelden)
        
        # Coördinaten (eerste 4 elementen)
        coords_true = y_true[:, :4]
        coords_pred = y_pred[:, :4]
        
        # Confidence score (laatste element)
        conf_true = y_true[:, 4]
        conf_pred = y_pred[:, 4]

        # 2. Bounding Box Coördinaten Loss (alleen op object-locaties)
        coord_loss_raw = self.mse(coords_true, coords_pred)
        # Gewogen met coord_weight en object_mask
        coord_loss = self.coord_weight * tf.reduce_sum(coord_loss_raw * tf.expand_dims(object_mask, axis=1), axis=1)

        # 3. Confidence Loss - Object
        # De loss op de confidence wanneer er WEL een object is
        object_confidence_loss = object_mask * self.mse(conf_true, conf_pred)
        
        # 4. Confidence Loss - No Object
        # De loss op de confidence wanneer er GEEN object is (gebruik een kleiner gewicht)
        no_object_confidence_loss = self.no_object_weight * no_object_mask * self.mse(conf_true, conf_pred)
        
        # 5. Totale Loss
        total_loss = coord_loss + object_confidence_loss + no_object_confidence_loss
        
        return tf.reduce_mean(total_loss)

if __name__ == '__main__':
    print("YoloLikeLoss klasse (verbeterde versie) gedefinieerd.")
    print("Je kunt nu ook negatieve (zonder marker) voorbeelden toevoegen aan je data!")
