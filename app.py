
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('/content/fruit_classifier.keras')

# Fruit classes
class_names = ['Apple', 'Avocado', 'Banana', 'Blackberry', 'Papaya']

st.title("🍎 Fruit Classification")
st.write("Upload a fruit image to identify it.")

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    index = np.argmax(prediction[0])
    confidence = prediction[0][index] * 100

    st.success(f"Predicted Fruit: {class_names[index]}")
    st.info(f"Confidence: {confidence:.2f}%")
