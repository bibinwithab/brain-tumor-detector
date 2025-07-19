from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = ".\brain-tumor-detector\backend\Brain Tumor MRI"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  
)

train_generator = train_datagen.flow_from_directory(
    './Brain Tumor MRI/Training',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
print("Train generator class indices:", train_generator.class_indices)

val_generator = train_datagen.flow_from_directory(
    './Brain Tumor MRI/Training',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
print("Validation generator class indices:", val_generator.class_indices)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    './Brain Tumor MRI/Testing',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
print("Test generator class indices:", test_generator.class_indices)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator, 
    epochs=25,
    validation_data=val_generator,
)

loss, acc = model.evaluate(test_generator)
print(f"Test Loss: {loss}, Test Accuracy: {acc}")

model.save('./models/brain_tumor_detector_model.h5')