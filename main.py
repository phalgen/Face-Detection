import numpy as np
import cv2
import pickle

print(cv2)
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name":1}

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)

while(True):
    ret, frame =cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    if face_cascade.empty():
        print("Error loading cascade classifier.")
        break
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w] #(cord1-height,cord2-height)

        #recogniser
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)



        img_item="WhatsApp Image 2023-05-21 at 13.36.22.jpg"
        cv2.imwrite(img_item,roi_gray)

        color = (255, 0, 0)
        thickness = 2
        x_cord = x+w
        y_cord = y+h
        cv2.rectangle(frame, (x, y), (x_cord, y_cord), color, thickness)




    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
