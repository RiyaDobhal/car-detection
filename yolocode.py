import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


#load the model
net = cv.dnn.readNetFromDarknet('yolov2.cfg','yolov3_custom_final.weights')

#classes
classes = []
with open ('class.names','r') as f:
    classes=[line.strip() for line in f.readlines()]

#reading image
my_img = cv.imread('test1.jpg')
#resizing image
my_img=cv.resize(my_img,(1000,800))
#image dimensions
ht,wt,_=my_img.shape


blob=cv.dnn.blobFromImage(my_img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)

net.setInput(blob)

last_layer=net.getUnconnectedOutLayersNames()
layer_out=net.forward(last_layer)

boxes=[]
confidence=[]
class_ids=[]

for output in layer_out:
    for detection in output:
        score=detection[5:]
        class_id= np.argmax(score)
        confidence=score[class_id]
        if(confidence>.6):
            center_x=int(detection[0]*wt)
            center_y=int(detection[1]*ht)
            w=int(detection[2]*wt)
            h=int(detection[3]*ht)
            x=int(center_x - h/2)
            y=int(center_y - h/2)
            boxes.append([x,y,w,h])
            confidence.append((float(confidence)))
            class_ids.append(class_id)

indexes=cv.dnn.NMSBoxes(boxes,confidence,.5,.4)
font=cv.FONT_HERSHEY_PLAIN
colors=np.random.uniform(0,255,size=(len(boxes),3))
for i in indexes.flatten():
    x,y,w,h=boxes[i]
    label=str(classes(class_ids[i]))
    confidence=str(round(confidence[i],2))
    color=color[i]
    cv.rectangle(my_img,(x,y),(x+w),(y+h),color,2)
    cv.putText(my_img,label+" "+confidence,(x,y+20),font,2,(0,0,0),2)

cv.imshow('img',my_img)
cv.waitKey(0)
cv.destroyAllWindows()