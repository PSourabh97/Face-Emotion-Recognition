from keras.models import load_model   # To load my model 'model.h5'
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2                            # used to display an image in a window
import numpy as np

from flask import Flask, render_template, Response
import cv2
import numpy as np
app=Flask(__name__)
# At first we will load 'haarcascade_frontalface_default.xml'. This will responsible for finding the faces in the frame.
face_classifier = cv2.CascadeClassifier(r'C:\Users\user\Desktop\DEEP LREARNING PROJECT\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\user\Desktop\DEEP LREARNING PROJECT\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']  # All possible output expressions.

cap = cv2.VideoCapture(0)   # For initializing the webcam.



def gen_frames():
    while True:
     # Grab a single frame of video
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # To convert colour to Gray because our training data is in gray.
            faces = face_classifier.detectMultiScale(gray,1.3,5) # After opening the input webcam image should be scaled.

            for (x,y,w,h) in faces:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) # Setting the colour and thickness of the rectangle.
             roi_gray = gray[y:y+h,x:x+w]                                # To set rectangular box's Height, width, length.
             roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)  # Resizing to 48,48 because my model trained on 48,48



             if np.sum([roi_gray])!=0:
              roi = roi_gray.astype('float')/255.0  # Standerdizing the region of interest by decreasing the pixel size.
              roi = img_to_array(roi)               # My model trained on array so i am conerting into array.
              roi = np.expand_dims(roi,axis=0)

              prediction = classifier.predict(roi)[0]  # Now we will use the classifier for prediction.
              label=emotion_labels[prediction.argmax()] # To level the max predicted emotion.
              label_position = (x,y-10)
              cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)  # Setting the colour and thickness of font
             else:
              cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)       
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame') # This will take the frames from webcan and will send to 'index.html'
if __name__=='__main__':
    app.run(debug=True)
#if __name__=='__main__':
#    app.run(host='0.0.0.0',port=8080)