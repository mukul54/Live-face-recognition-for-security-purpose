import cv2
import numpy as np
import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
from pygame import mixer
import time

detector = dlib.get_frontal_face_detector()

FRmodel = load_model('model_weight.h5')
print("Total Params:", FRmodel.count_params())
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def alarm():
        mixer.init()
        mixer.music.load("/home/mukul/Desktop/machine_learning/iitg.ai/facenet/Wake-up-sounds.mp3")
        mixer.music.play()
        time.sleep(0.05)

        
def recognize_face(face_descriptor, database):
    encoding = img_to_encoding(face_descriptor, FRmodel)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' % (name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
            
    if int(identity) <=10 or int(identity) >= 31 and min_dist <=0.068:
        return str('Mukul: Welcome to the office'), min_dist
    
    if int(identity) >=11 and int(identity) <=20 and min_dist <=0.068:
        return str('Jayant: Welcome to the office'), min_dist
        
    if int(identity) >=21 and int(identity) <= 30 and min_dist <=0.068:
        return str('Chandan: Welcome to the office'), min_dist
        
    if  min_dist >=0.068 and min_dist <= 0.071:
    
         return str('Not Sure: You are not allowed'), min_dist
    
    if min_dist > 0.071:
        return str('Not a member'), min_dist



def extract_face_info(img, img_rgb, database):
    faces = detector(img_rgb)
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            name, min_dist = recognize_face(image, database)
            
            if min_dist < 0.071:
               cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
               cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
              cv2.putText(img, 'Not a member', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
              alarm()
              
              
def initialize():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = fr_utils.img_path_to_encoding(file, FRmodel)
    return database


def recognize():
    database = initialize()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if(img.size == 0 or img_rgb.size == 0):
            continue
                
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)
        for subject in subjects:
            extract_face_info(img, img_rgb, database)
        
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


recognize()
