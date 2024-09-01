import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import zipfile
import time

def load_images_from_zip(zip_path, label):
    images = []
    labels = []
    
    # Check if the ZIP file exists
    if not os.path.isfile(zip_path):
        print(f"File {zip_path} does not exist.")
        return images, labels

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    print(f"Processing {file_name}...")
                    with zip_ref.open(file_name) as file:
                        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                        if img is not None:
                            img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
                            images.append(img)
                            labels.append(label)
                        else:
                            print(f"Failed to load image: {file_name}")
    except zipfile.BadZipFile:
        print(f"File {zip_path} is not a valid ZIP file.")
    return images, labels

def plot_sample_images(images, labels, title="Sample Images"):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 9)):
        plt.subplot(3, 3, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color representation
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_sample_images_with_preds(images, labels, preds):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 9)):
        plt.subplot(3, 3, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color representation
        plt.title(f"True: {labels[i]}, Pred: {preds[i]}")
        plt.axis('off')
    plt.suptitle("Test Data with Predictions")
    plt.show()

# Paths to the ZIP files
train_zip = 'new_train.zip'
test_zip = 'new_test.zip'

# Timing the image loading and preprocessing
start_time = time.time()

# Load training images
cat_images_train, cat_labels_train = load_images_from_zip(train_zip, 0)  # Assuming all images in train.zip are cats
dog_images_train, dog_labels_train = load_images_from_zip(train_zip, 1)  # Assuming all images in train.zip are dogs

# Combine and shuffle the training data
images_train = np.array(cat_images_train + dog_images_train)
labels_train = np.array(cat_labels_train + dog_labels_train)

# Flatten the images
images_train = images_train.reshape(len(images_train), -1)

# Normalize pixel values
images_train = images_train / 255.0

# Load test images
cat_images_test, cat_labels_test = load_images_from_zip(test_zip, 0)  # Assuming all images in test.zip are cats
dog_images_test, dog_labels_test = load_images_from_zip(test_zip, 1)  # Assuming all images in test.zip are dogs

# Combine and shuffle the test data
images_test = np.array(cat_images_test + dog_images_test)
labels_test = np.array(cat_labels_test + dog_labels_test)

# Flatten the test images
images_test = images_test.reshape(len(images_test), -1)

# Normalize pixel values
images_test = images_test / 255.0

end_time = time.time()
print(f"Time taken for loading and preprocessing: {end_time - start_time} seconds")

# Visualize the training data
plot_sample_images(images_train.reshape(-1, 64, 64, 3), labels_train, title="Training Data Sample")

# Train the SVM model
start_time = time.time()

svm = SVC(kernel='linear')  # You can also experiment with other kernels like 'rbf'
svm.fit(images_train, labels_train)

end_time = time.time()
print(f"Time taken for training the SVM model: {end_time - start_time} seconds")

# Predict on test data
start_time = time.time()

y_pred = svm.predict(images_test)

end_time = time.time()
print(f"Time taken for making predictions: {end_time - start_time} seconds")

# Evaluate the performance
print("Accuracy:", accuracy_score(labels_test, y_pred))
print("Classification Report:\n", classification_report(labels_test, y_pred))

# Visualize some sample images with predictions
plot_sample_images_with_preds(images_test.reshape(-1, 64, 64, 3), labels_test, y_pred)
