from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import numpy as np
import os
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import cv2
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import keras
import pickle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("An Efficient Fruit Identification and Ripening Detection Using CNN Algorithm")
main.geometry("1300x900")

global classifier
global filename
global X,Y

def upload():
  global filename
  global X,Y
  filename = filedialog.askdirectory(initialdir = ".")
  X = np.load("model/images.txt.npy")
  Y = np.load("model/labels.txt.npy")
  Y = to_categorical(Y)
  img = X[5].reshape(256,256,3)
  cv2.imshow('Sample Train Image',cv2.resize(img,(450,450)))
  cv2.waitKey(0)
  X = X.astype('float32')
  X = X/255
  indices = np.arange(X.shape[0])
  np.random.shuffle(indices)
  X = X[indices]
  Y = Y[indices]

  text.delete('1.0', END)
  text.insert(END,filename+' train images dataset Loaded\n')
  pathlabel.config(text=filename+" loaded")
  

def loadModel():
    global X,Y
    global classifier
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
          loaded_model_json = json_file.read()
          classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        pathlabel.config(text="Neural Network Model Generated Successfully")
        text.delete('1.0', END)
        text.insert(END,'Neural Network Model Generated Successfully. See black console for CNN layers\n')
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print(classifier.summary())
        train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1.0/255.)
        train_generator = train_datagen.flow_from_directory('dataset', batch_size = 20, class_mode = 'categorical', target_size = (64, 64))
        validation_generator = test_datagen.flow_from_directory('dataset', batch_size = 20, class_mode = 'categorical', target_size = (64, 64))
        hist = classifier.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
        classifier.save_weights('model/weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
          json_file.write(model_json)
        pathlabel.config(text="Neural Network Model Generated Successfully See black console for CNN layers")
        text.delete('1.0', END)
        text.insert(END,'Neural Network Model Generated Successfully. See black console for CNN layers')
    predict = classifier.predict(X)
    predict = np.argmax(predict, axis=1)
    Y = np.argmax(Y, axis=1)
    cnf_matrix = confusion_matrix(Y, predict)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    text.delete('1.0', END)
    text.insert(END,"Fruit Detection FPR : "+str(np.amax(FPR)*100)+"%\n")
    text.insert(END,"Fruit Detection TPR : "+str(np.amax(TPR)*100)+"%\n")
    text.insert(END,"Fruit Detection TNR : "+str(np.amax(TNR)*100)+"%\n")
    text.insert(END,"Fruit Detection FNR : "+str(np.amax(FNR)*100)+"%\n")
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('MTCNN Accuracy & Loss Graph')
    plt.show()

def ripeDetection(frame, lower, upper,start, end):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower = np.array([161, 155, 84])
    #upper = np.array([179, 255, 255])

    mask = cv2.inRange (hsv, lower, upper)
    contours,temp = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
      if len(contours[i]) > 10:
          red_area = contours[i] 
          x, y, w, h = cv2.boundingRect(red_area)
          print(str(x)+" "+str(y)+" "+str(start)+" "+str(end))
          x = start
          y = end
          img = frame[y:(y+h),x:(x+w)]
          cv2.putText(frame, 'Ripe', (x, y+20),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)

          
    
def classification():
    name = filedialog.askopenfilename(initialdir="testImages")
    pathlabel.config(text=name+" loaded")
    frame = cv2.imread(name)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 120])
    upper = np.array([15, 255, 255])

    mask = cv2.inRange (hsv, lower, upper)
    contours,temp = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = 0
    start_x = 0
    start_y = 0
    for i in range(len(contours)):
      if len(contours[i]) > 10:
          red_area = contours[i] 
          x, y, w, h = cv2.boundingRect(red_area)
          img = frame[y:(y+h),x:(x+w)]
          img = cv2.resize(img, (256,256))
          im2arr = np.array(img)
          im2arr = im2arr.reshape(1,256,256,3)
          XX = np.asarray(im2arr)
          XX = XX.astype('float32')
          XX = XX/255
          preds = classifier.predict(XX)
          predict = np.argmax(preds)
          print(predict)
          if predict == 1:
            start_x = x
            start_y = y
            cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 2)
            cv2.putText(frame, 'Fruit', (x, y),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
            ripeDetection(frame, np.array([22, 93, 0]), np.array([45, 255, 255]),start_x,start_y)            
            output = 1
    if output == 1:
        ripeDetection(frame, np.array([161, 155, 84]), np.array([179, 255, 255]),start_x,start_y)
        #ripeDetection(frame, np.array([22, 93, 0]), np.array([45, 255, 255]),start_x,start_y)
    frame = cv2.resize(frame,(500,500))    
    cv2.imshow("classification Result",frame)
    cv2.waitKey(0)
    

def exit():
    global main
    main.destroy()
  

font = ('times', 16, 'bold')
title = Label(main, text='An Efficient Fruit Identification and Ripening Detection Using CNN Algorithm',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
loadButton = Button(main, text="Upload Fruit Train Images Dataset", command=upload)
loadButton.place(x=50,y=200)
loadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=250)


uploadButton = Button(main, text="Generate & Load MTCNN Model", command=loadModel)
uploadButton.place(x=50,y=300)
uploadButton.config(font=font1)

uploadButton = Button(main, text="Upload Test Image & Fruit Detection", command=classification)
uploadButton.place(x=50,y=350)
uploadButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

text=Text(main,height=20,width=70)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=200)
text.config(font=font1) 

main.config(bg='chocolate1')
main.mainloop()
