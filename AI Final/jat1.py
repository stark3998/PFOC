from shutil import *
from PIL import Image,ImageTk
import cv2
import os
import numpy as np
import speech_recognition as sr
from DocumentRetrievalModel import DocumentRetrievalModel as DRM
from ProcessedQuestion import ProcessedQuestion as PQ
import re
import sys


mic_name = "Microsoft Sound Mapper - Input"
sample_rate = 48000
chunk_size = 2048
r = sr.Recognizer()
t=[]
mic_list = sr.Microphone.list_microphone_names()
for i, microphone_name in enumerate(mic_list):
    if microphone_name == mic_name:
        device_id = i

def get_audio():
    with sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size) as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
         
        try:
            text = r.recognize_google(audio)
            return text
     
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
     
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))



user="1"
folder="1"
def detect():
    user=input("Enter User Name :")
    folder=user
    sampleNum = 0
    facecascade = cv2.CascadeClassifier('face.xml')
    eyecascade = cv2.CascadeClassifier('eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y + h, x:x + h], (500, 500))
            cv2.imwrite("Images/" + user + "." + str(folder) + "." + str(sampleNum) + ".jpg", f)
            count += 1
            cv2.waitKey(200)
        cv2.imshow("camera : ", frame)
        cv2.waitKey(1)
        if sampleNum > 75:
            break
    camera.release()
    cv2.destroyAllWindows()

def trainer():
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    path="C:\\Users\\Stark\\Desktop\\PFOC\\AI Final\\Images"
    def getimageIds(path):
        imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        Ids = []
        for imagepath in imagepaths:
            faceimg = Image.open(imagepath).convert('L')
            facenp = np.array(faceimg, 'uint8')
            Id = int(os.path.split(imagepath)[-1].split('.')[1])
            faces.append(facenp)
            print(Id)
            Ids.append(Id)
            cv2.imshow('Training the dataset', facenp)
            cv2.waitKey(10)
        return Ids, faces

    Ids, faces = getimageIds(path)
    recogniser.train(faces, np.array(Ids))
    recogniser.write('recogniser/recogniser_all.yml')
    cv2.destroyAllWindows()

def recognise():
    path = "C:\\Users\\Stark\\Desktop\\PFOC\\AI Final\\Images"
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    path = "C:\\Users\\Stark\\Desktop\\PFOC\\AI Final\\Images"
    for imagepath in imagepaths:
        ID = int(os.path.split(imagepath)[-1].split('.')[1])
    facecascade = cv2.CascadeClassifier('face.xml')
    eyecascade = cv2.CascadeClassifier('eye.xml')
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read('C:\\Users\\Stark\\Desktop\\PFOC\\AI Final\\recogniser\\recogniser_all.yml')
    camera = cv2.VideoCapture(0)
    sampleNum = 0
    folder = '0'
    font = cv2.FONT_HERSHEY_DUPLEX
    conf=0
    ret,frame=camera.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        while(folder=='0' and conf<40.0):
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            folder,conf = rec.predict(gray[y:y + h, x:x + w])
            print(folder," - ",conf)
    camera.release()
    cv2.destroyAllWindows()
    return(str(folder))

def recognise1():
    path = "C:\\Users\\Stark\\Desktop\\PFOC\\AI Final\\Images"
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    path = "C:\\Users\\Stark\\Desktop\\PFOC\\AI Final\\Images"
    for imagepath in imagepaths:
        ID = int(os.path.split(imagepath)[-1].split('.')[1])
    facecascade = cv2.CascadeClassifier('face.xml')
    eyecascade = cv2.CascadeClassifier('eye.xml')
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read('C:\\Users\\Stark\\Desktop\\PFOC\\AI Final\\recogniser\\recogniser_all.yml')
    camera = cv2.VideoCapture(0)
    sampleNum = 0
    folder = '0'
    font = cv2.FONT_HERSHEY_DUPLEX
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            folder,conf = rec.predict(gray[y:y + h, x:x + w])
            cv2.putText(frame, str(folder)+"  "+str(conf), (x, y + h), font, 2, 255)
            f = cv2.resize(gray[y:y + h, x:x + h], (1000, 1000))
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


def readd():
    paragraphs = []
    datasetFile = open("dataset/943.txt","r")
    for para in datasetFile.readlines():
        if(len(para.strip()) > 0):
            paragraphs.append(para.strip())


def main():
    print("Bot> Please wait, Loading preferences from Home Base")
    #trainer()
    name=recognise()
    #datafile=input("Input Data File Name : ")
    try:
        datasetName="dataset/"+name+".txt"
        print(datasetName)
    except FileNotFoundError:
        print("Bot> Oops! I am unable to locate \"" + datasetName + "\"")

    print("Bot> Hey! I am ready. Ask me factoid based questions only :P")
    print("Bot> You can say me Bye anytime you want")
    greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$",re.IGNORECASE)
    isActive = True
    response="Hello User"
    while isActive:
        name=recognise()
        if(name!='0'):
            datasetName="dataset/"+name+".txt"
            print(datasetName)
            try:
                datasetFile = open(datasetName,"r")
            except FileNotFoundError:
                print("Bot> Oops! I am unable to locate \"" + datasetName + "\"")
                datasetFile = open("dataset/1.txt","r")
                #exit()
            paragraphs = []
            try:
                for para in datasetFile.readlines():
                    if(len(para.strip()) > 0):
                        paragraphs.append(para.strip())
                drm = DRM(paragraphs,True,True)
            except:
                print("Bot> Oops! Error in reading dataset")
                continue
            userQuery = input("You> ")
            if(not len(userQuery)>0):
                print("Bot> You need to ask something")
            elif greetPattern.findall(userQuery):
                response = "Hello!"
            elif userQuery.strip().lower() == "bye":
                response = "Bye Bye!"
                isActive = False
            elif userQuery.strip().lower() == "train faces":
                trainer()
            elif userQuery.strip().lower() == "new":
                detect()
            elif userQuery.strip().lower() == "train":
                trainer()
            elif userQuery.strip().lower() == "new user":
                detect()
                trainer()
            elif userQuery.strip().lower() == "detect faces":
                recognise1()
            elif userQuery.strip().lower() == "speech":
                try:
                    userQuery=get_audio()
                except:
                    print("Error Recognizing Audio")
                    continue
                print("User Said : ",userQuery)
                if greetPattern.findall(userQuery):
                    response="Hello!"
                else:
                    print(name," said : ",userQuery)
                    pq=PQ(userQuery,True,False,True)
                    response=drm.query(pq)
            else:
                pq = PQ(userQuery,True,False,True)
                response =drm.query(pq)
            print("Bot>",response)

if __name__=="__main__":
    readd()
