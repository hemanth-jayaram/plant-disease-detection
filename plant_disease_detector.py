#!/usr/bin/env "C:\Users\heman\AppData\Local\Programs\Python\Python311\python.exe"

import sys
import os

# Check if we're using the correct Python version
if sys.executable.lower().find('python311') == -1:
    print("Error: Please run this script with Python 3.11.8")
    print("Current Python version:", sys.executable)
    print("Please use the correct Python interpreter")
    sys.exit(1)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from sklearn.utils import class_weight

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def setup_data():
    # Use the existing dataset directory
    data_dir = r'C:\Users\heman\Plant disease detection Dataset'
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset directory not found at {data_dir}")
    
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def build_model():
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Unfreeze more layers for better feature extraction
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Add more complex custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile the model with a slightly higher learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

def train_model(model, train_generator, validation_generator):
    if train_generator.samples == 0:
        print("No training data found! Please add images to both 'Healthy' and 'Diseased' folders.")
        return None
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Add callbacks for better training
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model for more epochs
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=50,  # Increased from 20 to 50
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping]
    )
    
    return history

def plot_training_history(history):
    # Plot training & validation accuracy
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Set up data generators
    train_generator, validation_generator = setup_data()
    
    # Build and train the model
    model = build_model()
    history = train_model(model, train_generator, validation_generator)
    
    if history is not None:
        # Plot training history
        plot_training_history(history)
        
        # Save the model
        model.save('plant_disease_detector.h5')
        
        # Example prediction (uncomment and provide your test image path)
        # predict_image(model, 'path_to_your_test_image.jpg')
    else:
        print("No training was performed due to lack of data.")
