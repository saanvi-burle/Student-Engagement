### Database needed
- postgres@PostgreSQL 16

### Packages need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow
- pip install flask

### dataset used
- data 
- which included train and test files

### Train and Test Emotion detector
- with data/train dataset
- python file: TrainEmotionDetector.py TestEmotionDetector.py

Took 3-4 hours for i7 11gen processor with 16 GB RAM.
after Training , trained model structure and weights are stored in the project directory named as.
model/emotion_model.json
model/emotion_model.h5

### website
- templates/index.html
- static folder contains css and js files 

### run project
- python app.py or flask run
- if video needs to be changed add video in directory and update line 42 in app.py with video name with extension
- if live camera capture is expected, comment line 42 and uncomment line 47 in app.py


### flow
- train model
- test model
- run app.py or flask
- website runs on port 5000 (http://127.0.0.1:5000)
- click view details button for any facilitator 
- video or camera opens and detects emotion with green square
- press q to quit or exit
- as soon as you exit, analysis is shown on page
