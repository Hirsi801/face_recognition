import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

def load_dataset(data_dir):
    """
    Load face dataset from directory structure:
    data_dir/
        person1/
            img1.jpg
            img2.jpg
            ...
        person2/
            img1.jpg
            ...
    """
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            label_dict[current_label] = person_name
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:  # Only process if image is valid
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (160, 160))  # ResNet50 input size
                    faces.append(img)
                    labels.append(current_label)
            current_label += 1
    
    faces = np.array(faces, dtype='float32')
    labels = np.array(labels)
    
    # Normalize pixel values to [-1, 1]
    faces = (faces - 127.5) / 127.5
    
    return faces, labels, label_dict

def split_dataset(faces, labels, test_size=0.2, val_size=0.1):
    """
    Split dataset into train, validation, and test sets
    """
    # First split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        faces, labels, test_size=test_size, stratify=labels, random_state=42)
    
    # Then split train+val into train and val
    val_size_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_relative, stratify=y_trainval, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_datasets(X_train, X_val, y_train, y_val, batch_size=32):
    """Prepare TensorFlow datasets with augmentation"""
    # Data augmentation pipeline
  # In data_preparation.py
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.25),  # Increased rotation range
        layers.RandomZoom(0.25),      # Increased zoom range
        layers.RandomBrightness(0.3), # Increased brightness variation
        layers.RandomContrast(0.2),
        layers.GaussianNoise(0.02),   # Added noise
        layers.RandomTranslation(0.1, 0.1)  # Added translation
    ])
    
    def preprocess_train(image, label):
        image = augmentation(image)
        return image, label
    
    # Training dataset with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=1024)
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset without augmentation
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def save_dataset(X_train, X_val, X_test, y_train, y_val, y_test, label_dict, save_dir='processed_data'):
    """Save processed dataset to disk"""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    # Save label dictionary
    with open(os.path.join(save_dir, 'label_dict.txt'), 'w') as f:
        for label, name in label_dict.items():
            f.write(f"{label}:{name}\n")

def load_processed_data(save_dir='processed_data'):
    """Load processed dataset from disk"""
    X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(save_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(save_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
    
    # Load label dictionary
    label_dict = {}
    with open(os.path.join(save_dir, 'label_dict.txt'), 'r') as f:
        for line in f:
            label, name = line.strip().split(':')
            label_dict[int(label)] = name
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_dict

if __name__ == '__main__':
    # Example usage
    data_dir = 'data'  # Directory with person folders containing images
    faces, labels, label_dict = load_dataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(faces, labels)
    save_dataset(X_train, X_val, X_test, y_train, y_val, y_test, label_dict)
    print("Dataset prepared and saved successfully!")