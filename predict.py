import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from sys import argv
from keras.models import load_model
from keras.preprocessing.image import load_img , img_to_array

def print_help_message():
    print("""
        Enter type and file name
        e.g. test.py image /abosulte_path/to/img
             test.py video /abosulte_path/to/video
             test.py live [0 : webcam or 1 : external]
        """)

if len(argv) < 2:
    print_help_message()
    quit()

# load ssd model to predict faces (annotations)
cvNet = cv2.dnn.readNetFromCaffe('model/architecture.txt','model/weights.caffemodel')
print("loaded ssd model")

#load prediction model
model = load_model("C:/Users/abhis/jupyter/Untitled Folder/mask_prediction.h5")
print("loaded weights of model")

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
assign = {'0':'Mask','1':"No Mask"}

def processImage(image):
    image =  adjust_gamma(image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    for i in range(0, detections.shape[2]):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                im = cv2.resize(frame,(124,124))
                im = np.array(im)/255.0
                im = im.reshape(1,124,124,3)
                result = model.predict(im)
                if result>0.5:
                    label_Y = 1
                else:
                    label_Y = 0
                boundary_size = int(image.shape[1]/300)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), boundary_size)
                text_size = int(image.shape[1]/700)
                cv2.putText(image,assign[str(label_Y)] , (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (36,255,12), text_size)
        except:
            print("error occured while prediction")
            pass
    return image

if argv[1] == "image":
    if os.path.exists(argv[2]):
        image =  cv2.imread(argv[2])
        image = processImage(image)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print ("File does not exist")
        print_help_message()

elif argv[1] == "video":
    if os.path.exists(argv[2]):
        cam = cv2.VideoCapture(argv[2]) #0=front-cam, 1=back-cam
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)
        while True:
            response , image = cam.read()
            if response == False:
                break
            image = processImage(image)
            img = cv2.resize(image,(960, 540))
            cv2.imshow("Face mask detection (press q to quit)", img)

            ## press q or Esc to quit    
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                break

    cam.release()
    cv2.destroyAllWindows()

elif argv[1] == "live":
    cam = cv2.VideoCapture(int(argv[2])) #0=front-cam, 1=back-cam
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)
    while True:
        ret, image = cam.read()
        image = processImage(image)
        cv2.imshow("Face mask detection (press q to quit)", image)

        ## press q or Esc to quit    
        if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
            break

    cam.release()
    cv2.destroyAllWindows()

else:
    print (argv[1], " is not a correct type")
    print_help_message()