import cv2
import os
import matplotlib.pyplot as plt 
img = cv2.imread('C:/Progs and Concepts/AI and ML intern/Project/images/car/Image_1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
path='C:/Progs and Concepts/AI and ML intern/Project/images/car'
i=1
for filename in os.listdir(path):
    print(filename)
    img = cv2.imread(os.path.join(path,filename))
    print(img.shape)
    cv2.imshow('car'+str(i)+'image'+,img)
    i+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()