import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

train_dir = 'D:\\Certifications\\TensorFlow\\Coursera\\Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning\\Week 4\\horse-or-human'
test_dir = 'D:\\Certifications\\TensorFlow\\Coursera\\Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning\\Week 4\\validation-horse-or-human'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=test_generator,
    validation_steps=8
)
