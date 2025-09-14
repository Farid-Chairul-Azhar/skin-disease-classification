import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from PIL import Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    """
    Membuat heatmap Grad-CAM untuk model dan input gambar.

    Args:
        img_array (ndarray): Input gambar ukuran (1, 224, 224, 3)
        model (tf.keras.Model): Model klasifikasi CNN
        last_conv_layer_name (str): Nama layer konvolusi terakhir
        pred_index (int): Index kelas target (optional)

    Returns:
        heatmap (ndarray): Array 2D heatmap Grad-CAM
        predictions (ndarray): Output prediksi model
    """
    if last_conv_layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Layer '{last_conv_layer_name}' tidak ditemukan dalam model.")

    grad_model = Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap ke gambar asli.

    Args:
        heatmap (ndarray): Heatmap Grad-CAM (2D)
        image (PIL.Image): Gambar asli
        alpha (float): Transparansi heatmap
        colormap: Colormap dari OpenCV

    Returns:
        PIL.Image: Gambar hasil overlay heatmap
    """
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    image_np = np.array(image)
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]  # Buang alpha channel
    elif image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)

    overlayed_img = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlayed_img)
