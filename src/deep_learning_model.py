# ============================================
# 📁 File: src/deep_learning_model.py
# 🎯 Mục đích: Huấn luyện mô hình học sâu (1D CNN)
# ============================================

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def train_cnn_model(X_train, y_train, X_test, y_test, num_classes, model_dir="data/models"):
    os.makedirs(model_dir, exist_ok=True)

    # One-hot encode
    y_train_cat = to_categorical(y_train - 1, num_classes=num_classes)
    y_test_cat = to_categorical(y_test - 1, num_classes=num_classes)

    # Kiến trúc CNN1D
    model = Sequential([
        Conv1D(64, 5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(2),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callback
    checkpoint_path = os.path.join(model_dir, "cnn_best.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    # Huấn luyện
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Đánh giá
    y_pred = model.predict(X_test).argmax(axis=1) + 1
    print("\n=== 🧠 CNN Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    model.save(os.path.join(model_dir, "cnn_final.h5"))
    print(f"✅ Mô hình CNN đã lưu tại {model_dir}/")

    history_df = pd.DataFrame(history.history)
    history_file_path = os.path.join(model_dir, "training_history.csv")
    history_df.to_csv(history_file_path, index=False)
    print(f"✅ Lịch sử huấn luyện đã lưu tại {history_file_path}")

    return history
