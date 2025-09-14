import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import cv2
import os
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input

#MODEL LOADING 
@st.cache_resource(show_spinner="Memuat model deteksi kulit") 
def load_model():
    model_path = "models/best_vgg16_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di: `{model_path}`")
        st.stop()

    try:
        # Load full model (not just weights)
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model:\n\n```\n{str(e)}\n```")
        st.stop()

    class_names = ["Melanoma", "Psoriasis"]
    return model, class_names

model, CLASS_NAMES = load_model()

#Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image)
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]
    overlayed_img = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlayed_img)

#Streamlit UI Setup
st.set_page_config(page_title="Deteksi Kulit AI", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
    .title {text-align: center;font-weight: bold;font-size: 42px;color: #003366;margin-bottom: 20px;}
    .footer {text-align: center;font-size: 12px;color: gray;margin-top: 50px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Deteksi Kulit AI")
    st.markdown("### Aplikasi ini mengenali:")
    st.markdown("- Melanoma\n- Psoriasis")
    st.caption("Dibuat oleh Doni Rijki Ramadan © 2025")

st.markdown("<div class='title'>Deteksi Otomatis Penyakit Kulit</div>", unsafe_allow_html=True)

#Panduan
with st.expander(" Panduan Penggunaan Aplikasi"):
    st.markdown("""
### Langkah-langkah:
1. **Upload** gambar lesi kulit (.jpg/.png).
2. Masukkan data pasien.
3. Aplikasi akan menampilkan prediksi dan visualisasi Grad-CAM.
4. Anda bisa mengunduh hasil diagnosis.

### Catatan:
- Gambar harus **fokus dan jelas**.
- Aplikasi **tidak menggantikan diagnosa medis**.
""")

#Form Pasien
st.markdown("### Informasi Pasien")
col1, col2, col3 = st.columns(3)
with col1:
    nama_pasien = st.text_input("Nama Pasien", value="Tidak diketahui")
with col2:
    usia_pasien = st.number_input("Usia Pasien", min_value=0, max_value=120, value=0)
with col3:
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Tidak diketahui", "Laki-laki", "Perempuan"])

#Upload Gambar
st.markdown("### Upload Gambar Lesi Kulit")
uploaded_file = st.file_uploader("Upload file (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Pratinjau Gambar", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    pred_index = int(np.argmax(predictions))
    pred_label = CLASS_NAMES[pred_index]
    confidence = float(predictions[0][pred_index]) * 100

    st.success("Gambar berhasil diproses.")
    st.markdown("### Hasil Prediksi")
    st.markdown(f"**Hasil**: {pred_label}")
    st.markdown(f"**Keyakinan**: {confidence:.2f}%")
    st.progress(confidence / 100)

    #Confidence Chart
    st.markdown("### Confidence per Kelas")
    fig, ax = plt.subplots()
    bars = ax.bar(CLASS_NAMES, predictions[0], color=["#cc3300", "#339933"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Distribusi Keyakinan Model")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center')
    st.pyplot(fig)

    #Grad-CAM
    if st.button("Tampilkan Grad-CAM"):
        with st.spinner("Menghasilkan Grad-CAM..."):
            heatmap = make_gradcam_heatmap(img_array, model)
            alpha = st.slider("Transparansi Heatmap", 0.0, 1.0, 0.4, step=0.05)
            overlay_img = overlay_heatmap(heatmap, img, alpha=alpha)
            st.image(overlay_img, caption="Visualisasi Grad-CAM", use_container_width=True)

    #Penjelasan
    st.markdown("### Penjelasan Penyakit")
    if pred_label == "Melanoma":
        st.warning("Melanoma adalah kanker kulit ganas. Segera konsultasikan dengan dokter.")
    else:
        st.info("Psoriasis adalah kondisi kulit kronis. Dapat dikendalikan dengan pengobatan.")

    #Diagnosa TXT 
    txt_content = (
        "Hasil Diagnosa Deteksi Kulit AI\n"
        "===============================\n\n"
        f"Nama Pasien     : {nama_pasien}\n"
        f"Usia            : {usia_pasien} tahun\n"
        f"Jenis Kelamin   : {jenis_kelamin}\n\n"
        "Prediksi Lesi Kulit:\n"
        f"- Jenis Penyakit    : {pred_label}\n"
        f"- Tingkat Keyakinan : {confidence:.2f}%\n\n"
        "Catatan:\n"
        f"{'Melanoma adalah kanker kulit. Segera konsultasi ke dokter.' if pred_label.lower() == 'melanoma' else 'Psoriasis adalah peradangan kulit kronis yang tidak menular.'}\n\n"
        "Informasi ini tidak menggantikan diagnosa medis resmi.\n"
    )

    st.download_button(
        label="Download Hasil Diagnosa (TXT)",
        data=txt_content,
        file_name="hasil_diagnosa_kulit.txt",
        mime="text/plain"
    )

    #Simpan Riwayat
    st.session_state.history.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": pred_label,
        "confidence": confidence,
        "image": img.copy(),
        "nama": nama_pasien,
        "usia": usia_pasien,
        "gender": jenis_kelamin
    })

#Riwayat Diagnosa
if st.session_state.history:
    st.markdown("---")
    st.subheader("Riwayat Diagnosa")
    for idx, item in enumerate(reversed(st.session_state.history[-100:]), 1):
        with st.expander(f"Hasil ke-{idx}"):
            st.image(item["image"], width=300, caption="Gambar Lesi")
            st.write(f"Waktu         : **{item.get('timestamp', 'Tidak diketahui')}**")
            st.write(f"Nama          : **{item['nama']}**")
            st.write(f"Usia          : **{item['usia']}** tahun")
            st.write(f"Jenis Kelamin : **{item['gender']}**")
            st.write(f"Prediksi      : **{item['label']}**")
            st.write(f"Keyakinan     : **{item['confidence']:.2f}%**")

    if st.button("Hapus Seluruh Riwayat"):
        st.session_state.history = []
        st.success("Riwayat berhasil dihapus.")

#Footer
st.markdown("<div class='footer'>© 2025 | Deteksi Kulit AI - VGG16 | Farid Chairul Azhar</div>", unsafe_allow_html=True)
