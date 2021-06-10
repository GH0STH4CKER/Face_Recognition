import face_recognition as fr
from cv2 import cv2
import os
#from tkinter import Tk
#Tk().withdraw()

img = fr.load_image_file('images/4in1.jpg')

enc = fr.face_encodings(img)

def encode_face(folder) :
    encoding_list = []

    for filename in os.listdir(folder) :
        known_img = fr.load_image_file(f'{folder}{filename}')
        known_enc = fr.face_encodings(known_img)[0]
        encoding_list.append((known_enc,filename))

    return encoding_list

def find_face() :
    face_loc = fr.face_locations(img)

    for person in encode_face('images/') :
        enc_face = person[0]
        filename = person[1] 

        is_target_face = fr.compare_faces(enc_face,enc,tolerance=0.55)
        print(f'{is_target_face} {filename}')

        if face_loc:
            face_no = 0
            for location in face_loc :
                if is_target_face[face_no] :
                    label = filename
                    create_frame(location,label)
                face_no += 1

def create_frame(location,label) :
    top,right,bottom,left = location

    label = label.replace('.png','')
    label = label.replace('.jpg','')
    
    cv2.rectangle(img,(left,top),(right,bottom),(26, 237, 30),2)
    cv2.rectangle(img,(left,bottom+20),(right,bottom),(26, 237, 30),-1)
    cv2.putText(img,label,(left+3,bottom+14),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

def render_image() :
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow('Face Recognition',rgb_img)
    cv2.waitKey(0)

find_face()
render_image()
                    