import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from utilities import focal_tversky, tversky_loss, tversky

# Función para cargar modelo desde JSON


def cargar_modelo(json_path):
    with open(json_path, "r") as json_file:
        modelo_json = json_file.read()
    modelo = model_from_json(modelo_json)
    return modelo


# Cargar modelos una vez al inicio del script
try:
    model_clasificacion = cargar_modelo("resnet-50-MRI.json")
    model_segmentacion = cargar_modelo("ResUNet-MRI.json")
except Exception as e:
    st.error(f"Error cargando los modelos: {e}")

# Redimensionar imagen una sola vez


def resize_image(image):
    return image.resize((256, 256))


def preprocess_image(image):
    # Redimensionar imagen al tamaño deseado (256x256)
    image_resized = resize_image(image)
    # Convertir imagen a un array numpy
    image_array = np.array(image_resized)
    # Asegurarse de que la imagen tenga 3 canales (RGB)
    if image_array.ndim == 2:
        # Si la imagen es en blanco y negro, convertirla a RGB duplicando el canal
        image_array = np.stack((image_array,) * 3, axis=-1)
    # Normalizar valores de píxeles
    image_array = image_array / 255.0
    return image_array


def predecir_clasificacion(imagen, model):
    resultado = model.predict(np.expand_dims(imagen, axis=0))
    clase_predicha = "Sin Tumor" if resultado[0][0] > 0.5 else "Con Tumor"
    return clase_predicha


def predecir_segmentacion(imagen, model):
    resultado = model.predict(np.expand_dims(imagen, axis=0))
    mascara_predicha = (resultado > 0.5).astype(np.uint8)[0]
    return mascara_predicha


def main():
    st.title("Detección de Tumores Cerebrales")

    uploaded_file = st.file_uploader(
        "Cargar Imagen", type=["jpg", "png", "jpeg", "tif"])

    if uploaded_file is not None:
        try:
            imagen = Image.open(uploaded_file)
            imagen_array = preprocess_image(imagen)
            st.image(imagen, caption='Imagen cargada', use_column_width=True)

            clase = predecir_clasificacion(imagen_array, model_clasificacion)
            st.write(f"Clasificación: {clase}")

            if clase == "Con Tumor":
                mascara = predecir_segmentacion(
                    imagen_array, model_segmentacion)
                # Asegurarse de que la máscara tenga valores binarios
                mascara = np.where(mascara > 0.5, 1, 0)
                st.image(mascara, caption='Máscara de Segmentación',
                         use_column_width=True)
        except Exception as e:
            st.error(f"Error procesando la imagen: {e}")


if __name__ == "__main__":
    main()
