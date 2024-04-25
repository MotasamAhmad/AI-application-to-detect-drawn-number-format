import os
import tensorflow as tf
from tensorflow.keras import layers, models

# تحميل بيانات MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# تحويل صيغة الصور
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# تحويل العلامات إلى تصنيفات واحدة
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# تحديد المسار الكامل لحفظ النموذج
model_path = os.path.join(os.getcwd(), 'model.h5')

# بناء نموذج CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation


='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# تجميع النموذج
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# حفظ النموذج
model.save(model_path)
