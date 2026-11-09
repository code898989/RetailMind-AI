import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

MODEL_PATH = os.path.join("models", "product_resnet50_v7.keras")

def build_transfer_model(num_classes: int = 5, input_shape=(224, 224, 3)):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="RetailMind_ResNet50_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model

if __name__ == "__main__":
    model, base = build_transfer_model(num_classes=5)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[SUCCESS] ResNet50 Transfer model saved at: {MODEL_PATH}")
