import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    """
    Menghasilkan heatmap Grad-CAM dari gambar input untuk visualisasi perhatian model.

    Args:
        img_array (np.ndarray): Gambar preprocessed bentuk (1, 224, 224, 3)
        model (tf.keras.Model): Model klasifikasi
        last_conv_layer_name (str): Nama layer konvolusi terakhir
        pred_index (int): Index kelas target (jika None → ambil yang tertinggi)

    Returns:
        heatmap (np.ndarray): Array heatmap Grad-CAM 2D
        predictions (np.ndarray): Output prediksi model
    """
    # Cek layer tersedia
    if last_conv_layer_name not in [l.name for l in model.layers]:
        conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("Model tidak memiliki layer konvolusi.")
        last_conv_layer_name = conv_layers[-1]

    # Bangun model gradcam
    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Hitung gradien dan mean
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Hitung heatmap
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (H, W)

    # Normalisasi heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap if max_val == 0 else heatmap / max_val

    return heatmap.numpy(), predictions.numpy()
