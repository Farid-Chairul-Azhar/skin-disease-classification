import streamlit as st

def show_advice(pred_label):
    """
    Menampilkan rekomendasi awal berdasarkan label prediksi.
    
    Args:
        pred_label (str): Label kelas prediksi, misalnya 'Melanoma' atau 'Psoriasis'
    """
    st.markdown("### 🩺 Rekomendasi Awal")

    label = str(pred_label).strip().lower()

    if label == "melanoma":
        st.warning("🔴 **Kemungkinan besar ini adalah *Melanoma*.** Segera konsultasi dengan dokter spesialis kulit untuk pemeriksaan lanjutan.")
        st.markdown("""
        - 🏥 **Periksa ke fasilitas kesehatan dalam waktu 1–2 hari.**
        - 🌞 Hindari paparan sinar UV secara langsung.
        - 📸 Dokumentasikan perubahan ukuran/warna lesi.
        """)
    elif label == "psoriasis":
        st.info("ℹ️ **Ini kemungkinan adalah *Psoriasis*.** Walau tidak menular, kondisi ini bisa kambuh dan memerlukan perawatan rutin.")
        st.markdown("""
        - 💧 Gunakan pelembap kulit secara teratur.
        - 😌 Kurangi stres dan jaga pola hidup sehat.
        - 👨‍⚕️ Konsultasikan ke dokter jika gejala berulang atau meluas.
        """)
    else:
        st.info("📋 Tidak ada saran khusus karena label tidak dikenali.")
