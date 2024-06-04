import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score

# Função para padronizar as imagens
def padronizar_imagem(imagem_caminho):
    imagem = cv2.imread(imagem_caminho, cv2.IMREAD_GRAYSCALE)
    imagem = cv2.resize(imagem, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    return imagem

# Diretórios de treino e teste
faces_path_treino = "imagens/treino/"
faces_path_teste = "imagens/teste/"

# Listar arquivos de imagens
lista_faces_treino = [f for f in os.listdir(faces_path_treino) if os.path.isfile(os.path.join(faces_path_treino, f))]
lista_faces_teste = [f for f in os.listdir(faces_path_teste) if os.path.isfile(os.path.join(faces_path_teste, f))]

# Preparar dados de treinamento
dados_treinamento, sujeitos = [], []

for i, arq in enumerate(lista_faces_treino):
    imagem_path = os.path.join(faces_path_treino, arq)
    imagem = padronizar_imagem(imagem_path)
    dados_treinamento.append(imagem)
    sujeito = arq.split('_')[1]
    sujeitos.append(int(sujeito))

dados_treinamento = np.asarray(dados_treinamento)
sujeitos = np.asarray(sujeitos, dtype=np.int32)

# Treinar modelo LBPH
modelo_lbph = cv2.face.LBPHFaceRecognizer_create()
modelo_lbph.train(dados_treinamento, sujeitos)

# Salvar o modelo treinado
modelo_path = "modelos/modelo_lbph.yml"
modelo_lbph.save(modelo_path)
