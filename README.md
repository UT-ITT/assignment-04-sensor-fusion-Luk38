[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion
## 1. Perspective Transformation
starten mit:

```
python image_extractor.py --input_path INPUT_PATH --output_path OUTPUT_PATH --resolution WIDTH HEIGHT
```
Für den output path auch genau den Dateinamen mit Endung eingeben. Beispiel:
```
python image_extractor.py --input_path perspective_transformation/sample_image.jpg --output_path sample_image_warped.jpg --resolution 800 600
```
- ESC Änderungen Rückgängig machen
- S Neues Bild unter den Output path speichern
- Q Programm Fenster schließen und Programm beenden.

## 2. AR Game
Das Board am besten flach auf den Tisch legen, mit der Hand von der näheren Seite in das Feld.

Mit [Mediapipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#video) Landmarker wird der Zeigefinger getrackt.

Mit dem Zeigefinger kann man dann den Schläger frei im Bild bewegen und muss verhindern, dass der Ball zum unteren Bildrand gelangt.

Pro Abpraller am oberen Bildrand bekommt man einen Punkt, wenn der Ball den unteren berührt wird der Score zurückgesetzt und man muss von 0 anfangen.
Man kann das Fenster und Programm mit Q oder ESC beenden.

Es kann vorkommen dass eine falsche Hand erkannt wird, weil vom original frame und nicht vom gewarpten getrackt wird. Um das zu beheben sollte man die Hände aus der Kamera nehmen und neu hinhalten.

Wenn man außerhalb vom unteren Bordrand seinen finger bewegt, kann man den Schläger ganz am Rand unten bewegen.