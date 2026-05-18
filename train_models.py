import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
import numpy as np
import time
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score

# ============================================================
# AUTO-CREATE FOLDERS (Para hindi mag-error kapag nag-save)
# ============================================================
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # More epochs = better convergence

# ============================================================
# LOAD AND PREPARE DATASET
# ============================================================
print("=" * 60)
print("LOADING DATASET: horses_or_humans")
print("=" * 60)

dataset, info = tfds.load(
    "horses_or_humans",
    as_supervised=True,
    with_info=True
)

print(f"Total examples: {info.splits['train'].num_examples}")

# Split: 80% train, 20% test
train_raw = dataset["train"]
train_size = int(0.8 * info.splits["train"].num_examples)

train_data_raw = train_raw.take(train_size)
test_data_raw = train_raw.skip(train_size)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing
train_data = train_data_raw.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data_raw.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Dataset prepared successfully!\n")

# ============================================================
# MODEL BUILDING FUNCTION
# ============================================================
def build_and_train_model(base_model_class, model_name, input_shape=(224, 224, 3)):
    """Build, compile, train, and evaluate a transfer learning model."""
    
    print("=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    
    # Load pre-trained base model (without top layers)
    base_model = base_model_class(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base model layers
    
    # Build the full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")  # Binary classification
    ])
    
    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train and measure time
    start_time = time.time()
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS
    )
    training_time = time.time() - start_time
    
    # Get final training and validation accuracy from history
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_data)
    
    # Detailed predictions for classification report
    y_pred_probs = model.predict(test_data)
    y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)
    y_true = np.concatenate([y for x, y in test_data], axis=0).reshape(-1)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Training Accuracy:   {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Training Time:       {training_time:.2f} seconds")
    print("\nClassification Report:")
    # Updated target names for Horses vs Humans
    print(classification_report(y_true, y_pred, target_names=["horse", "human"]))
    
    # Save the model
    save_path = os.path.join("saved_models", f"{model_name.lower()}_model.keras")
    model.save(save_path)
    print(f"Model saved to: {save_path}\n")
    
    return {
        "Model": model_name,
        "Training Accuracy": round(train_acc, 4),
        "Validation Accuracy": round(val_acc, 4),
        "Test Accuracy": round(test_acc, 4),
        "Training Time (s)": round(training_time, 2)
    }, history


# ============================================================
# TRAIN ALL THREE MODELS
# ============================================================
results = []
histories = {}

# 1. MobileNetV2
result1, hist1 = build_and_train_model(MobileNetV2, "MobileNetV2")
results.append(result1)
histories["MobileNetV2"] = hist1

# 2. ResNet50
result2, hist2 = build_and_train_model(ResNet50, "ResNet50")
results.append(result2)
histories["ResNet50"] = hist2

# 3. EfficientNetB0
result3, hist3 = build_and_train_model(EfficientNetB0, "EfficientNetB0")
results.append(result3)
histories["EfficientNetB0"] = hist3

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Save to CSV
df.to_csv("results/comparison.csv", index=False)
print("\nResults saved to results/comparison.csv")

# ============================================================
# PLOT TRAINING CURVES
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, hist) in enumerate(histories.items()):
    axes[idx].plot(hist.history['accuracy'], label='Train Accuracy')
    axes[idx].plot(hist.history['val_accuracy'], label='Val Accuracy')
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Accuracy')
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
print("Training curves saved to results/training_curves.png")
# plt.show()  # Commented out para hindi mag-block sa terminal