import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn

STEERING_CORRECTION = 0.1
EPOCHS = 5 
DATA_LOC = "data"

samples = []
with open(DATA_LOC + '/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for sample in reader:
    samples.append(sample)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# use a generator so we don't attempt to load all samples into memory at once
def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      
      images=[]
      angles=[]
      for batch_sample in batch_samples:
        center_name = DATA_LOC + '/IMG/' + batch_sample[0].split('/')[-1]
        left_name = DATA_LOC + '/IMG/' + batch_sample[1].split('/')[-1]
        right_name = DATA_LOC + '/IMG/' + batch_sample[2].split('/')[-1]
        center_image = cv2.imread(center_name)
        left_image = cv2.imread(left_name)
        right_image = cv2.imread(right_name)
        center_angle = float(batch_sample[3])
        left_angle = center_angle + STEERING_CORRECTION # use left and right camera images with steering correction
        right_angle = center_angle - STEERING_CORRECTION
        images.append(center_image)
        images.append(left_image)
        images.append(right_image)
        angles.append(center_angle)
        angles.append(left_angle)
        angles.append(right_angle)
   
      augmented_images, augmented_angles = [], []
      for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image, 1)) # augment image/steering data by flipping images over y axis to reduce bias
        augmented_angles.append(angle*-1.0)

      X_train = np.array(augmented_images)
      y_train = np.array(augmented_angles)
      yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

height, width, channels = 160, 320, 3

model = Sequential()
# normalize the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(height, width, channels),output_shape=(height, width, channels)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop out irrelevant image data (sky and hood of car)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu")) # next few lines of model.add implement the nvidia model mentioned in lecture
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
	validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=EPOCHS, verbose=1)

# plot out a loss diagram
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
fig = plt.gcf()
fig.savefig('loss.png')

model.save('model.h5')
exit()
