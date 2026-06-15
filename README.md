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

Pro Abpraller am oberen Bildrand bekommt man einen Punkt, wenn der Ball den unteren Rand berührt wird der Score zurückgesetzt und man muss von 0 anfangen.
Man kann das Fenster und Programm mit Q oder ESC beenden.

Es kann vorkommen dass eine falsche Hand erkannt wird, weil vom original frame und nicht vom gewarpten getrackt wird. Um das zu beheben sollte man die Hände aus der Kamera nehmen und neu hinhalten.

Wenn man außerhalb vom unteren Bordrand seinen finger bewegt, kann man den Schläger ganz am Rand unten bewegen.

## 3. Sensor Fusion
Das Rechteck innerhalb der Marker wird wie in der vorherigen Aufgabe erkannt und gewarped.

Wenn man das Smartphone mit dem fünften Marker in die Kamera hält, wird der Marker mit ID 5 erkannt und ein roter Kreis darauf gezeichnet.
Außerdem muss man mit dem DIPPID Device an Port 5700 senden, sodass basierend auf den accelerometerdaten eine position auf dem bord vorhergesagt wird, auf die ein grüner Kreis gezeichnet wird.

Mit Button 1 in der Dippid App kann man die vorhersage reseten. 

Mit der rechten Pfeiltaste den alpha Wert erhöhen, mit der linken verringern. Der aktuelle Alpha-Wert wird im Fenster angezeigt.

Das Fenster und Programm lässt sich mit Q oder ESC beenden.

Zum Alpha-Wert: 

- Bei kleinem Alpha = 0.1 wird mehr auf die Kamera vertraut und die Vorhersage durch Accelerometer ist nicht stark, sodass der grüne Punkt dem roten viel schneller folgt.
- Bei großem Alpha = 0.95 ist die Vorhersage sehr stark und der grüne Punkt kann abdriften, da er hauptsächlich durch die Accelerometerdaten gestuert wird.
- Bei Alpha = 1.0 werden nur die Accelerometerdaten genommen und der grüne Punkt orrientiert sich garnicht mehr an dem Roten sondern nur an der Bewegung des DIPPID device.

