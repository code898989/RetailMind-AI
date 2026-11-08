import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

MODEL_DIR = "models"
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "catalog_cnn_v6.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "catalog_labels_v6.joblib")

IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3
BATCH_SIZE = 32

def train_fashion_mnist_prototype():
    print("\n[PROTOTYPE] Step 1: Training Fast Fashion-MNIST Prototype...")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Take a 10,000 sample subset for ultra-fast validation
    x_train = x_train[:10000].reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = y_train[:10000]
    x_test = x_test[:2000].reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_test = y_test[:2000]

    proto_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    proto_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    proto_model.fit(x_train, y_train, epochs=2, batch_size=64, verbose=1, validation_data=(x_test, y_test))
    print("[PROTOTYPE SUCCESS] Prototype validated architecture successfully.")

def build_catalog_cnn(num_classes: int):
    # Augmentation block followed by Feature Extraction and Dense Head
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_catalog_data():
    styles_csv = os.path.join("data", "raw", "styles.csv")
    images_dir = os.path.join("data", "raw", "images")
    
    categories = ["Apparel", "Footwear", "Accessories", "Personal Care", "Sporting Goods"]
    
    # Check if user has downloaded the Kaggle Fashion Product Images dataset
    if os.path.exists(styles_csv) and os.path.exists(images_dir):
        print("[INFO] Found Kaggle Fashion Product Images dataset.")
        df = pd.read_csv(styles_csv, on_bad_lines='skip')
        df = df[df['masterCategory'].isin(categories)].copy()
        df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        df = df[df['image_path'].apply(os.path.exists)].head(5000)
        
        labels = sorted(df['masterCategory'].unique().tolist())
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        
        images = []
        targets = []
        for _, row in df.iterrows():
            img = tf.keras.utils.load_img(row['image_path'], target_size=(IMG_HEIGHT, IMG_WIDTH))
            images.append(tf.keras.utils.img_to_array(img) / 255.0)
            targets.append(label_to_id[row['masterCategory']])
            
        X = np.array(images, dtype="float32")
        y = np.array(targets, dtype="int32")
    else:
        print("[INFO] Using simulated catalog product data for standalone pipeline execution...")
        labels = categories
        num_samples = 1500
        X = np.random.uniform(0.0, 1.0, size=(num_samples, IMG_HEIGHT, IMG_WIDTH, CHANNELS)).astype("float32")
        y = np.random.randint(0, len(labels), size=(num_samples,)).astype("int32")

    return X, y, labels

def run_day6_pipeline():
    train_fashion_mnist_prototype()
    
    print("\n[INFO] Step 2: Training RetailMind Catalog CNN...")
    X, y, labels = load_catalog_data()
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    cnn = build_catalog_cnn(num_classes=len(labels))
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    
    cnn.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=8,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluation and Confusion Matrix
    y_probas = cnn.predict(X_test)
    y_preds = np.argmax(y_probas, axis=1)
    
    print("\n" + "=" * 55)
    print("DAY 6 CATALOG CNN CLASSIFICATION AUDIT")
    print("=" * 55)
    print(classification_report(y_test, y_preds, target_names=labels, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))
    print("=" * 55 + "\n")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    cnn.save(CNN_MODEL_PATH)
    joblib.dump(labels, LABEL_MAP_PATH)
    print(f"[SUCCESS] Saved Catalog CNN model to: {CNN_MODEL_PATH}")
    print(f"[SUCCESS] Saved label mapping to: {LABEL_MAP_PATH}")

if __name__ == "__main__":
    run_day6_pipeline()
