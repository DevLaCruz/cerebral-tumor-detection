import json

# Función para verificar el contenido del archivo JSON
def verificar_json(json_path):
    with open(json_path, "r") as json_file:
        try:
            model_data = json.load(json_file)
            if "model" in model_data and "weights" in model_data:
                print(f"El archivo {json_path} contiene tanto la arquitectura como los pesos.")
            else:
                print(f"El archivo {json_path} solo contiene la arquitectura del modelo o está mal formateado.")
        except json.JSONDecodeError:
            print(f"El archivo {json_path} está vacío o no es un JSON válido.")

# Verificar los archivos JSON
verificar_json("resnet-50-MRI.json")
verificar_json("ResUNet-MRI.json")