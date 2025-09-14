import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input

def show_upload_section(model, CLASS_NAMES):
    st.markdown("### 📤 Upload Gambar Lesi Kulit")
    
    uploaded_file = st.file_uploader("Unggah file gambar (.jpg / .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Validasi file gambar
            img = Image.open(uploaded_file)
            img.verify()  # validasi image
            img = Image.open(uploaded_file).convert("RGB")  # load ulang karena verify() merusak file handle
            st.image(img, caption="Pratinjau Gambar", use_column_width=True)

            # Preprocessing untuk VGG16
            img_resized = img.resize((224, 224))
            img_array = keras_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prediksi
            predictions = model.predict(img_array)
            pred_index = int(np.argmax(predictions))
            pred_label = CLASS_NAMES[pred_index]
            confidence = round(float(predictions[0][pred_index]) * 100, 2)

            st.success("✅ Gambar berhasil diproses dan diprediksi.")
            return uploaded_file, img, pred_label, confidence, predictions

        except UnidentifiedImageError:
            st.error("❌ File yang diunggah bukan gambar yang valid.")
        except Exception as e:
            st.error(f"❌ Gagal memproses gambar atau prediksi: {e}")

    return None, None, None, None, None
