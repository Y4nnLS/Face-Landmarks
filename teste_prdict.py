import os
import cv2
import json
import numpy as np

# Função para padronizar as imagens
def padronizar_imagem(imagem_caminho):
    imagem = cv2.imread(imagem_caminho, cv2.IMREAD_GRAYSCALE)
    imagem = cv2.resize(imagem, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    return imagem

# Carregar o mapeamento de sujeitos
with open("treinamento/modelos/sujeito_map.json", "r") as f:
    sujeito_map = json.load(f)

# Inverter o mapeamento para obter o nome do sujeito a partir do índice
sujeito_map_inverso = {idx: name for name, idx in sujeito_map.items()}

# Carregar o modelo LBPH treinado
modelo_path = "modelos/modelo_lbph.yml"
modelo_lbph = cv2.face.LBPHFaceRecognizer_create()
modelo_lbph.read(modelo_path)

# Diretório contendo as imagens para predição
faces_path_predicao = "predict/"

# Listar arquivos de imagens para predição
lista_faces_predicao = [f for f in os.listdir(faces_path_predicao) if os.path.isfile(os.path.join(faces_path_predicao, f))]

# Preparar dados para predição
dados_predicao = []

for arq in lista_faces_predicao:
    imagem_path = os.path.join(faces_path_predicao, arq)
    imagem = padronizar_imagem(imagem_path)
    dados_predicao.append((arq, imagem))

# Fazer predição para cada imagem
for arq, imagem in dados_predicao:
    predicao, confianca = modelo_lbph.predict(imagem)
    sujeito_nome = sujeito_map_inverso.get(predicao, "Desconhecido")
    print(f"Arquivo: {arq}, Predição: {sujeito_nome}, Confiança: {confianca}")
