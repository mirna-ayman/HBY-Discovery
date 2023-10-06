import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import Sequential

#from keras.optimizers import adam
  
from tensorflow.keras.optimizers import RMSprop


#from keras.optimizers import SGD


#img = image.load_img("basedata/training/White-Nudity/2.jpg")
#plt.imshow(img)


#cv2.imread("basedata/training/White-Nudity/2.jpg")
#plt.show()
training = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)

training_dataset = training.flow_from_directory('basedata/training', target_size
                                             = (200,200),
                                             batch_size = 4,
                                             class_mode = 'binary')

validation_dataset = validation.flow_from_directory('basedata/validation', target_size
                                             = (200,200),
                                             batch_size = 4,
                                             class_mode = 'binary')


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation= 'relu', input_shape =(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation= 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation= 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation= 'relu'),
                                    ##
                                    tf.keras.layers.Dense(1, activation= 'sigmoid')
                                    ])

model.compile(loss= 'binary_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics =['accuracy'])

model_fit = model.fit(training_dataset,
                      steps_per_epoch = 16,
                      epochs = 16,
                      validation_data = validation_dataset)

print(validation_dataset.class_indices)
dir_path = 'basedata/testing'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+ '//' +i, target_size = (200,200))
    #plt.imshow(img)
    #plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis = 0)
    images = np.vstack([X])

    val = model.predict(images)
    if val == 0:
        print(i+"-"+"naked")
    else:
        print(i+"-"+"not naked")
    
