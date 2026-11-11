import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight", "Septoria Leaf Spot"]
IMG_SIZE = 224

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    if os.path.exists("tomato_model.h5"):
        _model = tf.keras.models.load_model("tomato_model.h5")
        return _model

    # Build model (weights only, no training — user must train or provide .h5)
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    _model = Model(inputs=base.input, outputs=out)
    _model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return _model

def preprocess(image_array):
    """Resize and normalize a NumPy image (H,W,3) to model input."""
    import cv2
    img = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_array):
    """
    Returns: (class_name: str, confidence: float 0-1, all_probs: list)
    image_array: RGB NumPy array (H, W, 3)
    """
    model = load_model()
    inp = preprocess(image_array)
    probs = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()


def train_model(data_dir, epochs=15, save_path="tomato_model.h5"):
    """
    Train on a PlantVillage-style directory structure:
        data_dir/
            Early Blight/  *.jpg
            Healthy/       *.jpg
            Late Blight/   *.jpg
            Septoria Leaf Spot/ *.jpg
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.7, 1.3],
    )

    train_data = train_gen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode="categorical", subset="training"
    )
    val_data = train_gen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode="categorical", subset="validation"
    )

    model = load_model()
    cb = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                             save_best_only=True, verbose=1)
    model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[cb])

    # Fine-tune last 20 layers
    base = model.layers[0]
    for layer in base.layers[-20:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[cb])
    print(f"Model saved to {save_path}")
