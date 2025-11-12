Gebruik:

1. Training van het Model (Op je ontwikkel-PC)
​Voer het trainingsscript uit. Dit leert het model de marker te herkennen en slaat de beste versie op als een bestand (best_marker_detector.h5).

python train_model.py

OPMERKING: Je moet de load_data functie in train_model.py aanpassen om jouw specifieke gelabelde data correct te laden en te verwerken.

2. Implementatie op de Drone (Deployment)
Kopieer het getrainde model (best_marker_detector.h5), drone_deploy.py en de benodigde Python-bibliotheken naar de onboard computer van de drone.
Zorg dat de drone is verbonden en de juiste DroneKit/MAVLink verbindingsstring (CONNECTION_STRING) in drone_deploy.py is ingesteld voor communicatie met de vluchtcontroller.

3.Start het AI-detectiesysteem op de drone:
<!-- end list -->

python drone_deploy.py


Het script zal de camera initialiseren, het AI-model inladen en beginnen met het berekenen van de visuele correcties op basis van de marker die het ziet.


4.Belangrijke Ontwikkelingspunten
Camera Kalibratie: De berekende pixel-afwijking van de marker moet nauwkeurig worden omgezet naar meters om de drone te corrigeren. Dit vereist een kalibratiefunctie in drone_deploy.py op basis van de actuele vlieghoogte.
Hardware Interface: De DroneKit/MAVLink commando's voor het verzenden van positionele correcties moeten volledig geïmplementeerd en getest worden in de main_drone_loop van drone_deploy.py.
<!-- end list -->
