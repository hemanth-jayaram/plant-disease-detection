# Plant Disease Detection Using Transfer Learning

This project implements a plant disease detection system using transfer learning with ResNet50. The model can classify plant leaves as either healthy or diseased.

## Features

- Transfer learning with pre-trained ResNet50 model
- Data augmentation for improved generalization
- Class weight handling for imbalanced datasets
- Visualization of training metrics
- Easy prediction interface

## Requirements

- Python 3.8+
- TensorFlow 2.19.0
- NumPy 1.24.3
- Matplotlib 3.7.1
- scikit-learn 1.3.0
- Pandas 2.0.3
- Seaborn 0.12.2

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your plant leaf images in the `Plant disease detection Dataset` directory under two subdirectories:
   - `Healthy` - for healthy plant leaves
   - `Diseased` - for diseased plant leaves

2. Run the training script:
```bash
python plant_disease_detector.py
```

3. After training, you can use the model to predict new images:
```python
from plant_disease_detector import predict_image
predict_image(model, 'path_to_your_test_image.jpg')
```

## Model Architecture

The model uses a pre-trained ResNet50 backbone with the following custom layers added:
- Global Average Pooling
- Dense layer with 512 units
- Dropout layer (0.5)
- Output layer with sigmoid activation

## Training Parameters

- Image size: 224x224
- Batch size: 32
- Learning rate: 0.00001
- Epochs: 20
- Validation split: 20%

## Results

The model provides:
- Training and validation accuracy plots
- Training and validation loss plots
- Saved model file (`plant_disease_detector.h5`)

## Notes

- Ensure your dataset is properly organized with clear class labels
- The model uses class weights to handle imbalanced datasets
- Adjust the learning rate and epochs as needed for your specific dataset
