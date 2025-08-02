import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from data_preparation import load_processed_data
from model import ArcFace, FaceRecognitionModel , l2_distance # Import the custom layer
import os

def evaluate_classification_model(model_path, X_test, y_test):
    """Evaluate the classification model"""
    # Load model with custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'ArcFace': ArcFace}
    )
    try:
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'ArcFace': ArcFace},
            compile=False  # Try without compiling first
        )
        
        # Recompile if needed
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print("Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('evaluation/classification_confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Possible solutions:")
        print("1. Make sure you've trained the model first")
        print("2. Verify the model path is correct")
        print("3. Check if the model file exists in the models directory")
        return None





def evaluate_siamese_model(model_path, X_test, y_test):
    """Evaluate the siamese model"""
    # Load model with custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'ArcFace':  ArcFace,  'l2_distance': l2_distance}
    )
    
    # Create test pairs
    X_pairs_test, y_pairs_test = create_pairs(X_test, y_test)
    
    # Predictions
    y_pred = model.predict([X_pairs_test[:, 0], X_pairs_test[:, 1]])
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_pairs_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('evaluation/siamese_roc_curve.png')
    plt.close()
    
    # Optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Binary predictions
    y_pred_binary = (y_pred > optimal_threshold).astype(int)
    
    # Classification report
    print("\nClassification Report at Optimal Threshold:")
    print(classification_report(y_pairs_test, y_pred_binary))
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"AUC: {roc_auc:.4f}")

def create_pairs(X, y):
    """Create positive and negative pairs for evaluation"""
    num_classes = len(np.unique(y))
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
    pairs = []
    labels = []
    
    # Create balanced pairs (equal number of positive and negative)
    n = min([len(indices) for indices in digit_indices])  # number of examples per class
    for d in range(num_classes):
        for i in range(n):
            # Positive pair
            z1, z2 = digit_indices[d][i], digit_indices[d][(i + 1) % n]
            pairs += [[X[z1], X[z2]]]
            labels += [1]
            
            # Negative pair (with different class)
            inc = (d + 1) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[inc][i]
            pairs += [[X[z1], X[z2]]]
            labels += [0]
    
    return np.array(pairs), np.array(labels)

if __name__ == '__main__':
    # Load processed data
    X_train, X_val, X_test, y_train, y_val, y_test, label_dict = load_processed_data()
    
    # Create evaluation directory
    os.makedirs('evaluation', exist_ok=True)
    
    # Evaluate classification model
    print("Evaluating classification model...")
    evaluate_classification_model('models/classification_model.keras', X_test, y_test)
    
    # Evaluate siamese model
    print("\nEvaluating siamese model...")
    evaluate_siamese_model('models/siamese_model.keras', X_test, y_test)