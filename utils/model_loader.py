@st.cache_resource(show_spinner="🔄 Memuat model deteksi kulit...")
def load_model():
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout

    model_path = "models/best_vgg16_model.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model tidak ditemukan di: `{model_path}`")
        st.stop()

    try:
        # ✅ Rekonstruksi arsitektur model
        input_tensor = Input(shape=(224, 224, 3))
        base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(2, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)

        # ✅ Load hanya weights-nya
        model.load_weights(model_path)

    except Exception as e:
        st.error(f"❌ Gagal memuat model:\n\n```\n{str(e)}\n```")
        st.stop()

    class_names = ["Melanoma", "Psoriasis"]
    return model, class_names
