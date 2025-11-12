# AI-Accurate-Drone-Dropping-Software-
Repository for the AI Accurate Drone Dropping Software 

# 🚁 Precisie Luchtdropping Systeem met AI 🎯

Dit project implementeert een Machine Learning (ML) model voor Computer Vision om de nauwkeurigheid van humanitaire voedselpakketdroppings via een drone te verbeteren. Het systeem corrigeert de GPS-locatie door visueel een vooraf geplaatste marker op de grond te detecteren.

## 🛠️ Benodigde Componenten

1.  **Hardware:** Drone met vluchtcontroller (bijv. Pixhawk/APM), Onboard Computer (bijv. Raspberry Pi 4 of Nvidia Jetson), en een USB/CSI-camera.
2.  **Software:** Linux-besturingssysteem, Python 3.8+, en de gespecificeerde bibliotheken.

## ⚙️ Installatie

### Stap 1: Omgeving Voorbereiden
Zorg ervoor dat Python 3 en `pip` geïnstalleerd zijn.

### Stap 2: Installeer Afhankelijkheden
Navigeer naar de projectmap en voer het volgende commando uit:

```bash
pip install -r requirements.txt

# 🚁 Drone AI Marker Detectie Systeem

Dit project implementeert een Convolutioneel Neuraal Netwerk (CNN) voor real-time objectdetectie (een marker) via een camera op een drone. De pipeline gebruikt een gespecialiseerde **Yolo-achtige Loss Functie** voor robuuste Bounding Box Regressie.

## 📁 Bestandsstructuur

Zorg voor de volgende mappen en bestanden:
drone_ai_project/
├── data/
│   └── raw_images/     # Hier komen al je ruwe afbeeldingen van de marker
├── requirements.txt
├── yolo_loss.py        # NIEUW: Custom Loss Functie
├── data_processing.py  # NIEUW: Data inlezen en splitsen
├── evaluate_model.py   # NIEUW: Model testen en visualiseren
├── model_architectuur.py
├── train_model.py
└── drone_deploy.py



---

## 🚀 Stap 1: Setup & Installatie

1.  **Installeer Vereisten:** Zorg dat je `requirements.txt` de volgende inhoud heeft en installeer de bibliotheken:

    ```bash
    # requirements.txt
    numpy
    tensorflow
    scikit-learn
    opencv-python
    pandas
    # dronekit (Optioneel, voor de echte drone)
    ```

    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Voorbereiding:**
    * Plaats al je marker-afbeeldingen in **`data/raw_images/`**.
    * Creëer het annotatiebestand **`data/annotations.csv`**. Dit CSV-bestand moet per rij één marker bevatten met **genormaliseerde** coördinaten (waarden tussen **0.0 en 1.0**): `filename,x_center,y_center,width,height`.

---

## 💻 Stap 2: De AI Pipeline Scripts

Kopieer en plak de code hieronder in de respectievelijke bestanden.

### A. `yolo_loss.py`

```python
import tensorflow as tf
from tensorflow.keras.losses import Loss

class YoloLikeLoss(Loss):
    """ Custom Loss Functie voor Bounding Box Regressie (zoals YOLO). """
    def __init__(self, coord_weight=5.0, no_object_weight=0.5, name='yolo_like_loss'):
        super().__init__(name=name)
        self.coord_weight = coord_weight
        self.no_object_weight = no_object_weight
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        object_mask = y_true[:, 4]        
        no_object_mask = 1.0 - object_mask 
        
        coords_true = y_true[:, :4]
        coords_pred = y_pred[:, :4]
        conf_true = y_true[:, 4]
        conf_pred = y_pred[:, 4]

        # 1. Coördinaten Loss (gewogen en alleen op object-locaties)
        coord_loss_raw = self.mse(coords_true, coords_pred)
        coord_loss = self.coord_weight * tf.reduce_sum(coord_loss_raw * tf.expand_dims(object_mask, axis=1), axis=1)

        # 2. Confidence Loss - Object & No-Object (gewogen)
        object_confidence_loss = object_mask * self.mse(conf_true, conf_pred)
        no_object_confidence_loss = self.no_object_weight * no_object_mask * self.mse(conf_true, conf_pred)
        
        # 3. Totale Loss
        total_loss = coord_loss + object_confidence_loss + no_object_confidence_loss
        
        return tf.reduce_mean(total_loss)