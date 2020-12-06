import os
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
bike_path='C:/Progs and Concepts/AI and ML intern/Project/images/motorcycle' #give your respective system file path here... the same applies for wherever the path is needed
car_path='C:/Progs and Concepts/AI and ML intern/Project/images/car'
test_img_path = 'C:/Progs and Concepts/AI and ML intern/Project/predict_images'
car_test_path = 'C:/Progs and Concepts/AI and ML intern/Project/predict_images/car1.jpg'
i=1
car_img_array =  []
bike_img_array = []
identify_arr1 = [1]*100
identify_arr2 = [2]*100
identify_arr = identify_arr1+identify_arr2
print(identify_arr)
print(len(identify_arr))
for filename in os.listdir(car_path):
    img = cv2.imread(os.path.join(car_path,filename))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(28,28))
    car_img_array.append(img)
for filename in os.listdir(bike_path):
    img = cv2.imread(os.path.join(bike_path,filename))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(28,28))
    bike_img_array.append(img)
img_array = car_img_array+bike_img_array
img_arr = np.array(img_array)
iden_arr = np.array(identify_arr)
print(len(img_arr))
x_train,x_test,y_train,y_test = train_test_split(img_arr,iden_arr,test_size=0.25)

print(x_train.shape)

y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

model = Sequential()

# COnvolutional Layer
model.add(Conv2D(filters = 32,kernel_size=(4,4),input_shape = (28,28,3),activation = 'softplus'))
# Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 32,kernel_size=(4,4),input_shape = (28,28,3),activation = 'softplus'))
# Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Flatten
model.add(Flatten())

# 128 Neurons in 1st Dense Layer
model.add(Dense(100,activation='softplus'))
model.add(Dense(60,activation='softplus'))

#Output Layer
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics = ['accuracy'])

print(model.summary())

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
print(len(y_cat_train))
print(y_cat_train)

model.fit(x_train,y_cat_train,epochs = 20) # batch_size =32

img_predict_arr = []
for filename in os.listdir(test_img_path):
    img = cv2.imread(os.path.join(test_img_path,filename))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(28,28))
    img_predict_arr.append(img)

img_test = cv2.imread(car_test_path)
img_test = cv2.resize(img_test,(28,28))
img_predict_array = np.array(img_predict_arr)
print(img_predict_array.shape)
plt.imshow(img_test)
(eval_loss, eval_accuracy) = model.evaluate(x_test,y_cat_test, batch_size=16,verbose=1)
img_class =  model.predict_classes(test_img)
print("The image is identified as",img_class)
#pred_output = model.prex
#print(pred_output)  
