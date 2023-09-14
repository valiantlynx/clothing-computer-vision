import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True
      


image = 3
# Define the callbacks
checkpoint_callback = ModelCheckpoint('my_model.h5', monitor='val_loss', save_best_only=True)
earlystop_callback = EarlyStopping(monitor='val_loss', patience=3)
tensorboard_callback = TensorBoard(log_dir='./logs')     
my_callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[image])
print(training_labels[image])
print(training_images[image])

#normalizing the data
training_images  = training_images / 255.0
test_images = test_images / 255.0

plt.imshow(training_images[image])
print(training_labels[image])
print(training_images[image])

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.Conv2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.Conv2D(2, 2),
                                    tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(training_images, training_labels, epochs=5)
model.fit(training_images, training_labels, validation_data=(test_images, test_labels),
          epochs=7, batch_size=32,
          callbacks=[my_callbacks, checkpoint_callback, earlystop_callback, tensorboard_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_loss, "\n", test_acc)

classifications = model.predict(test_images)

print('classifications[7] ', classifications[image]) #It's the probability that this item is each of the 10 classes
ans = test_labels[image] #this item


    
    




