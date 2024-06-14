import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score
import shutil
import random

# Função para padronizar as imagens
def padronizar_imagem(imagem_caminho):
    imagem = cv2.imread(imagem_caminho, cv2.IMREAD_GRAYSCALE)
    imagem = cv2.resize(imagem, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    return imagem

# Diretórios de treino e teste
faces_path_captured = "treinamento/imagens/captured_faces/"
faces_path_treino = "treinamento/imagens/treino/"
faces_path_teste = "treinamento/imagens/teste/"

if not os.path.exists(faces_path_treino):
    os.makedirs(faces_path_treino)

if not os.path.exists(faces_path_teste):
    os.makedirs(faces_path_teste)

# Organizar imagens capturadas em treino e teste
lista_faces_captured = [f for f in os.listdir(faces_path_captured) if os.path.isfile(os.path.join(faces_path_captured, f))]

for arq in lista_faces_captured:
    partes = arq.split('_')
    if len(partes) >= 2:
        sujeito = partes[0]
        numero = int(partes[2].split('.')[0])
        if numero <= 25:
            shutil.copyfile(os.path.join(faces_path_captured, arq), os.path.join(faces_path_treino, arq))
        else:
            shutil.copyfile(os.path.join(faces_path_captured, arq), os.path.join(faces_path_teste, arq))



# Listar arquivos de imagens
lista_faces_treino = [f for f in os.listdir(faces_path_treino) if os.path.isfile(os.path.join(faces_path_treino, f))]
lista_faces_teste = [f for f in os.listdir(faces_path_teste) if os.path.isfile(os.path.join(faces_path_teste, f))]

# Preparar dados de treinamento
dados_treinamento, sujeitos = [], []

for i, arq in enumerate(lista_faces_treino):
    partes = arq.split('_')
    if len(partes) >= 2:
        imagem_path = os.path.join(faces_path_treino, arq)
        imagem = padronizar_imagem(imagem_path)
        dados_treinamento.append(imagem)
        sujeito = partes[0]
        sujeitos.append(sujeito)

# Mapear nomes dos sujeitos para inteiros
unique_sujeitos = list(set(sujeitos))
sujeito_map = {name: idx + 1 for idx, name in enumerate(unique_sujeitos)}

# Salvar o mapeamento em um arquivo
import json
with open("treinamento/modelos/sujeito_map.json", "w") as f:
    json.dump(sujeito_map, f)

# Converter os nomes dos sujeitos para inteiros
sujeitos = np.array([sujeito_map[name] for name in sujeitos], dtype=np.int32)

# Treinar modelo LBPH
modelo_lbph = cv2.face.LBPHFaceRecognizer_create()
modelo_lbph.train(dados_treinamento, sujeitos)

# Verificar se o diretório de modelos existe, caso contrário, criar
modelo_dir = "modelos"
if not os.path.exists(modelo_dir):
    os.makedirs(modelo_dir)

# Salvar o modelo treinado
modelo_path = os.path.join(modelo_dir, "modelo_lbph.yml")
modelo_lbph.save(modelo_path)

# Preparar dados de teste
dados_teste, sujeitos_teste = [], []

for i, arq in enumerate(lista_faces_teste):
    partes = arq.split('_')
    if len(partes) >= 2:
        imagem_path = os.path.join(faces_path_teste, arq)
        imagem = padronizar_imagem(imagem_path)
        dados_teste.append(imagem)
        sujeito = partes[0]
        sujeitos_teste.append(sujeito)

sujeitos_teste = np.array([sujeito_map[name] for name in sujeitos_teste], dtype=np.int32)

# Avaliar modelo LBPH
y_pred_lbph = [modelo_lbph.predict(item)[0] for item in dados_teste]
acuracia_lbph = accuracy_score(sujeitos_teste, y_pred_lbph)

print("Acurácia do modelo LBPH:", acuracia_lbph)
