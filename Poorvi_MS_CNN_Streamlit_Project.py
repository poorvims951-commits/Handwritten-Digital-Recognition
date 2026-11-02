"""
CNN IMAGE CLASSIFICATION AND STREAMLIT APPLICATION
Student: Poorvi M.S
Batch: Artificial Intelligence September B

The application trains and evaluates a CNN on the MNIST dataset when it is
started for the first time. Training results are generated at runtime.
"""

# ============================================================
# PART A - CNN IMAGE CLASSIFICATION (MNIST)
# ============================================================

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CLASS_NAMES = [str(i) for i in range(10)]
EPOCHS = 5
BATCH_SIZE = 128


def build_cnn_model():
    """Create a CNN for classifying 28 x 28 grayscale digit images."""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(10, activation="softmax")
    ], name="MNIST_CNN")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


@st.cache_resource(show_spinner="Preparing MNIST data and training the CNN...")
def train_and_evaluate():
    """
    Load MNIST, preprocess the images, train the CNN and evaluate test data.
    Cached so the model is not retrained on every Streamlit interaction.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1] and add the grayscale channel.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Use a held-out portion of the training set for validation.
    x_fit, x_val = x_train[:-5000], x_train[-5000:]
    y_fit, y_val = y_train[:-5000], y_train[-5000:]

    model = build_cnn_model()
    history = model.fit(
        x_fit, y_fit,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    return {
        "model": model,
        "history": history.history,
        "x_test": x_test,
        "y_test": y_test,
        "predicted": predicted_labels,
        "probabilities": probabilities,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "train_count": int(len(x_train)),
        "test_count": int(len(x_test))
    }


def preprocess_uploaded_image(uploaded_file):
    """
    Convert an uploaded digit image to the MNIST input format.
    Supports common JPG/PNG images and handles light or dark backgrounds.
    """
    image = Image.open(uploaded_file).convert("L")
    image_array = np.asarray(image)

    # MNIST uses a dark background and a light digit. Invert if needed.
    corner_values = np.array([
        image_array[0, 0], image_array[0, -1],
        image_array[-1, 0], image_array[-1, -1]
    ])
    if np.mean(corner_values) > 127:
        image = ImageOps.invert(image)

    image = ImageOps.autocontrast(image)
    image_array = np.asarray(image)

    # Crop around the visible digit to reduce empty margins.
    mask = image_array > 30
    if np.any(mask):
        ys, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        image = image.crop((left, top, right, bottom))

    # Fit the digit inside a 20 x 20 area and centre it on a 28 x 28 canvas.
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2
    canvas.paste(image, (x_offset, y_offset))

    processed = np.asarray(canvas).astype("float32") / 255.0
    return np.expand_dims(processed, axis=(0, -1)), canvas


# ============================================================
# PART B - STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="Digit Recognition Studio",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Handwritten Digit Recognition Studio")
st.write(
    "Upload an image of a handwritten digit (0–9). "
    "The trained CNN analyses the image and predicts the most likely digit."
)
st.info(
    "Image tip: use one clearly written digit with good contrast. "
    "PNG and JPG images are supported."
)

with st.spinner("Loading the CNN and evaluating MNIST..."):
    results = train_and_evaluate()

model = results["model"]
history = results["history"]

tab_predict, tab_dashboard, tab_about = st.tabs(
    ["Digit Predictor", "CNN Results", "About the Project"]
)

with tab_predict:
    st.subheader("Try your own handwritten digit")
    uploaded = st.file_uploader(
        "Choose a digit image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image containing one handwritten digit."
    )

    if uploaded is not None:
        try:
            input_tensor, processed_image = preprocess_uploaded_image(uploaded)
            probabilities = model.predict(input_tensor, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities)) * 100

            left_col, right_col = st.columns([1, 2])
            with left_col:
                st.image(processed_image, caption="Processed 28 × 28 image",
                         width=180)
            with right_col:
                st.success(f"Predicted digit: {predicted_digit}")
                st.metric("Model confidence", f"{confidence:.2f}%")

            st.caption("Probability assigned to each digit")
            probability_chart = {
                "Digit": CLASS_NAMES,
                "Probability (%)": (probabilities * 100).round(2)
            }
            st.bar_chart(probability_chart, x="Digit", y="Probability (%)")
        except Exception as error:
            st.error(f"Unable to process this image. Please try another file. Details: {error}")
    else:
        st.warning("Upload an image to receive a prediction.")

with tab_dashboard:
    st.subheader("Dataset and model evaluation")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Training images", f"{results['train_count']:,}")
    metric2.metric("Test images", f"{results['test_count']:,}")
    metric3.metric("Test accuracy", f"{results['test_accuracy'] * 100:.2f}%")

    st.write("**Model architecture**")
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    st.code("\n".join(summary_lines), language="text")

    st.write("**Training and validation accuracy**")
    accuracy_data = {
        "Training accuracy": history["accuracy"],
        "Validation accuracy": history["val_accuracy"]
    }
    st.line_chart(accuracy_data)

    st.write("**Training and validation loss**")
    loss_data = {
        "Training loss": history["loss"],
        "Validation loss": history["val_loss"]
    }
    st.line_chart(loss_data)

    st.write("**Sample actual and predicted labels**")
    sample_count = min(12, len(results["x_test"]))
    sample_indices = np.linspace(0, len(results["x_test"]) - 1,
                                 sample_count, dtype=int)
    columns = st.columns(4)
    for position, index in enumerate(sample_indices):
        actual = int(results["y_test"][index])
        predicted = int(results["predicted"][index])
        with columns[position % 4]:
            st.image(results["x_test"][index].squeeze(),
                     caption=f"Actual: {actual} | Predicted: {predicted}",
                     width=110)
            if actual == predicted:
                st.caption("Correct prediction")
            else:
                st.caption("Incorrect prediction")

    incorrect = np.where(results["y_test"] != results["predicted"])[0]
    if len(incorrect) > 0:
        st.write("**Examples of incorrect predictions**")
        wrong_cols = st.columns(min(4, len(incorrect)))
        for position, index in enumerate(incorrect[:4]):
            with wrong_cols[position]:
                st.image(results["x_test"][index].squeeze(),
                         caption=(f"Actual: {int(results['y_test'][index])}, "
                                  f"Predicted: {int(results['predicted'][index])}"),
                         width=110)
    st.write(f"Test loss: **{results['test_loss']:.4f}**")

with tab_about:
    st.subheader("Project overview")
    st.write(
        "This project combines a Convolutional Neural Network (CNN) with a "
        "Streamlit interface. The CNN learns visual patterns from MNIST "
        "handwritten digit images. The application accepts a digit image, "
        "prepares it in the same 28 × 28 grayscale format and displays the "
        "predicted digit with its confidence."
    )
    st.write("**Dataset:** MNIST handwritten digits (0–9).")
    st.write("**Input:** One PNG or JPG image containing a handwritten digit.")
    st.write("**Output:** Predicted digit, confidence and class probabilities.")
    st.write("**Main tools:** Python, TensorFlow/Keras, NumPy, Pillow and Streamlit.")
    st.caption("Student: Poorvi M.S | Batch: Artificial Intelligence September B")
