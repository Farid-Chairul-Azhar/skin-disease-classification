import streamlit as st

def show_explanation(pred_label, debug=False):
    """
    Menampilkan penjelasan singkat terkait hasil prediksi kulit.

    Args:
        pred_label (str): Label prediksi, seperti 'Melanoma' atau 'Psoriasis'.
        debug (bool): Jika True, tampilkan log debug.
    """
    st.markdown("---")
    st.subheader("🧾 Penjelasan Singkat")

    label = str(pred_label).strip().lower()

    if debug:
        st.code(f"[DEBUG] Label prediksi diterima: {label}", language="python")

    if label == "melanoma":
        st.error("**Melanoma** adalah jenis kanker kulit ganas yang bisa menyebar dengan cepat jika tidak ditangani.")
        st.markdown("""
        - ⚠️ Warna gelap tidak merata, bentuk tidak simetris  
        - 🩺 Harus segera dikonsultasikan ke dokter spesialis kulit  
        """)
    elif label == "psoriasis":
        st.info("**Psoriasis** adalah peradangan kulit kronis yang menyebabkan sisik putih dan kemerahan.")
        st.markdown("""
        - ❗ Tidak menular, tapi dapat kambuh terus menerus  
        - 💊 Bisa dikontrol dengan pengobatan topikal atau terapi cahaya  
        """)
    else:
        st.warning("Label tidak dikenali. Tidak tersedia penjelasan untuk kondisi ini.")
