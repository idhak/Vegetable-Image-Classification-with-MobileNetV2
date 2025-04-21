# Proyek Klasifikasi Gambar - Vegetable Image Classification with MobileNetV2

This project is a deep learning-based image classification pipeline built using TensorFlow and Keras, trained to classify 15 types of vegetables using transfer learning with MobileNetV2. The dataset is obtained from Kaggle and is used to demonstrate a high-accuracy classification model.

## Dataset

The dataset used in this project is:
**[Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)**

It contains images of 15 different vegetables and is split into three folders:
- `train/`
- `validation/`
- `test/`

## Dependencies

```bash
pip install tensorflow tensorflowjs matplotlib seaborn scikit-image pandas opencv-python tqdm
```

## 🚀 Steps Performed

### 1. Data Preparation
- Upload Kaggle API key (`kaggle.json`) to download the dataset
- Extract and check dataset structure
- Count images in each folder (train, validation, test)
- Load class names

### 2. Data Preprocessing
- Resize images to 224x224
- Normalize pixel values
- Apply data augmentation on training set using `ImageDataGenerator`

### 3. Model Building
- Use `MobileNetV2` as base model (pre-trained on ImageNet)
- Add custom classification layers:
  - Conv2D + MaxPooling2D
  - GlobalAveragePooling
  - Dropout + Dense layers
- Use softmax output layer for multi-class classification
- Compile with Adam optimizer and categorical crossentropy

### 4. Training
- Train for up to 30 epochs
- Early stopping and dual-accuracy threshold callback to stop training if accuracy and val_accuracy > 97%

### 5. Evaluation & Visualization
- Evaluate accuracy on test set
- Plot training/validation accuracy and loss
- Show predictions on validation images
- Display confusion matrix
- Print classification report with precision, recall, and F1-score

### 6. Model Conversion
- Save model in:
  - HDF5 format (`.h5`)
  - TensorFlow SavedModel format
  - TensorFlow Lite format (`.tflite`)
  - TensorFlow.js format (`tfjs_model/`)

### 7. Inference (Optional)
- Load SavedModel for inference
- Sample prediction script provided for individual test image

## Results
- Achieved **100% test accuracy**
- High precision, recall, and F1-score across all classes
- Minimal overfitting due to transfer learning and augmentation

## Sample Classes
- Bean
- Bitter Gourd
- Bottle Gourd
- Brinjal
- Broccoli
- Cabbage
- Capsicum
- Carrot
- Cauliflower
- Cucumber
- Papaya
- Potato
- Pumpkin
- Radish
- Tomato

##  Notes
- Ensure `kaggle.json` is uploaded and placed correctly before running the script
- Trained on Google Colab with T4 GPU
- Model is ready for deployment on web or mobile apps using TFLite or TFJS

---

**Author:** *Idha Kurniawati*   
**Dataset Credit:** M Israk Ahmed (Kaggle)

