import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

def prepare_image(pil_img, target_size=(224, 224)):
    """
    Mengonversi gambar PIL menjadi array siap input model VGG16.

    Args:
        pil_img (PIL.Image.Image): Gambar dari file uploader.
        target_size (tuple): Ukuran target (default 224x224 untuk VGG16).

    Returns:
        np.ndarray: Gambar dalam format array 4D siap prediksi (1, 224, 224, 3).

    Raises:
        ValueError: Jika proses konversi gagal.
    """
    try:
        # ✅ Pastikan format RGB
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # ✅ Resize & ubah ke array float32
        img_resized = pil_img.resize(target_size)
        img_array = img_to_array(img_resized, dtype="float32")

        # ✅ Expand dims dan preprocessing
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        return img_array
    except Exception as e:
        raise ValueError(f"Gagal memproses gambar: {str(e)}")
