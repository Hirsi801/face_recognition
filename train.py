import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from model import FaceRecognitionModel
from data_preparation import load_processed_data , prepare_datasets
import os

# def prepare_datasets(X_train, X_val, y_train, y_val, batch_size=32):
#     """Prepare TensorFlow datasets with augmentation"""
#     # Data augmentation
#     augmentation = tf.keras.Sequential([
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.1),
#         layers.RandomBrightness(0.1),
#     ])
    
#     def preprocess_train(image, label):
#         image = augmentation(image)
#         return image, label
    
#     # Training dataset with augmentation
#     train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#     train_ds = train_ds.shuffle(buffer_size=1024)
#     train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
#     train_ds = train_ds.batch(batch_size)
#     train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
#     # Validation dataset without augmentation
#     val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
#     val_ds = val_ds.batch(batch_size)
#     val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
#     return train_ds, val_ds

def train_classification_model(X_train, X_val, y_train, y_val, num_classes, epochs=50):
    """Train the face classification model"""
    # Prepare datasets
    train_ds, val_ds = prepare_datasets(X_train, X_val, y_train, y_val)
    
    # Build model
    model = FaceRecognitionModel(num_classes=num_classes)
    classification_model = model.build_classification_model()
    
    # Compile model
    classification_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/classification_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            #  save_format='keras'  # Explicitly specify save format
        ),
        EarlyStopping(
            patience=10,
            monitor='val_accuracy',
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = classification_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history, classification_model

def train_siamese_model(X_train, X_val, y_train, y_val, num_classes, epochs=50):
    """Train the siamese verification model"""
    # Create positive and negative pairs
    X_pairs_train, y_pairs_train = create_pairs(X_train, y_train)
    X_pairs_val, y_pairs_val = create_pairs(X_val, y_val)
    
    # Build model
    model = FaceRecognitionModel(num_classes=num_classes)
    siamese_model = model.build_siamese_model()
    
    # Compile model
    siamese_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/siamese_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            # save_format='keras'  # Explicitly specify save format
        ),
        EarlyStopping(
            patience=10,
            monitor='val_accuracy',
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = siamese_model.fit(
        [X_pairs_train[:, 0], X_pairs_train[:, 1]],
        y_pairs_train,
        validation_data=([X_pairs_val[:, 0], X_pairs_val[:, 1]], y_pairs_val),
        batch_size=32,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history, siamese_model
def create_pairs(X, y):
    """Create positive and negative pairs for siamese training"""
    num_classes = len(np.unique(y))
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
    pairs = []
    labels = []
    
    for idx1 in range(len(X)):
        # Add a matching example
        x1 = X[idx1]
        y1 = y[idx1]
        idx2 = np.random.choice(digit_indices[y1])
        x2 = X[idx2]
        pairs += [[x1, x2]]
        labels += [1]
        
        # Add a non-matching example
        y2 = np.random.randint(0, num_classes)
        while y2 == y1:
            y2 = np.random.randint(0, num_classes)
        idx2 = np.random.choice(digit_indices[y2])
        x2 = X[idx2]
        pairs += [[x1, x2]]
        labels += [0]
    
    return np.array(pairs), np.array(labels)

if __name__ == '__main__':
    # Load processed data
    X_train, X_val, X_test, y_train, y_val, y_test, label_dict = load_processed_data()
    num_classes = len(label_dict)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train classification model
    print("Training classification model...")
    cls_history, cls_model = train_classification_model(
        X_train, X_val, y_train, y_val, num_classes, epochs=50
    )
    
    # Train siamese model
    print("\nTraining siamese model...")
    sia_history, sia_model = train_siamese_model(
        X_train, X_val, y_train, y_val, num_classes, epochs=30
    )
    
    print("Training completed! Models saved in 'models' directory.")