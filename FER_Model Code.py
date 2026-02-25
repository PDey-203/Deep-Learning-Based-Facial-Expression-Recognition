import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

train_ds = keras.utils.image_dataset_from_directory(
    directory=r"C:\Users\PRITAM\OneDrive\Desktop\train",
    image_size=(128, 128),
    batch_size=64,
    label_mode="int",
    color_mode="rgb",
)

val_ds = keras.utils.image_dataset_from_directory(
    directory=r"C:\Users\PRITAM\OneDrive\Desktop\validation",
    image_size=(128, 128),
    batch_size=64,
    label_mode="int",
    color_mode="rgb",
)
NUM_CLASSES = len(train_ds.class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


all_labels = np.concatenate([y.numpy() for x, y in train_ds])
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(all_labels), y=all_labels
)
class_weights = dict(enumerate(class_weights))


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ]
)

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = Sequential(
    [
        layers.Input(shape=(128, 128, 3)),
        data_augmentation,
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-3),
        ),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.2, patience=4, min_lr=1e-6
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    class_weight=class_weights,
    callbacks=callbacks,
)

save_path = r"C:\Users\PRITAM\OneDrive\Desktop\FER_Model.keras"
model.save(save_path)
print("Model saved at:", save_path)
