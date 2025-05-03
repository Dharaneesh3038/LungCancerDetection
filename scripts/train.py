import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

# Define constants
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
img_height, img_width = 224, 224
batch_size = 16
epochs = 10

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

# Function to load and split data
def load_and_split_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    all_data = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Extract images and labels
    images, labels = [], []
    for _ in range(len(all_data)):
        img_batch, label_batch = next(all_data)
        images.append(img_batch)
        labels.append(label_batch)
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    
    # Split dataset
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=42
    )
    valid_images, test_images, valid_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.67, stratify=temp_labels, random_state=42
    )
    
    return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)

# Load and split data
(train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = load_and_split_data(data_dir)

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = dict(enumerate(compute_class_weight(
    'balanced', classes=np.unique(np.argmax(train_labels, axis=1)), y=np.argmax(train_labels, axis=1)
)))

# Convert to tf.data.Dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# Rest of your train.py (model definition, training loop, etc.) remains the same
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # 3 classes: Benign, Malignant, Normal
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    class_weight=class_weights
)

for layer in base_model.layers[-10:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    class_weight=class_weights
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")

model.save('lung_cancer_resnet50_model.keras')
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('training_history.png')
plt.close()