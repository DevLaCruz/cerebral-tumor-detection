import tensorflow as tf
import numpy as np
import json

# Cargar el modelo desde el archivo HDF5
model = tf.keras.models.load_model('weights.hdf5', compile=False)

# Extraer la arquitectura del modelo en formato JSON
model_json = model.to_json()

# Extraer los pesos del modelo y convertirlos en listas
weights = model.get_weights()
weights_list = [w.tolist() for w in weights]

# Crear el diccionario que contendrá la arquitectura y los pesos
model_data = {
    "model": json.loads(model_json),  # convertir el string JSON a un diccionario
    "weights": weights_list
}

# Guardar el diccionario en un archivo JSON
with open('model_weights.json', 'w') as json_file:
    json.dump(model_data, json_file)
