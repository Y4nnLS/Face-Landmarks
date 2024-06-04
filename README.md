# Projeto de Reconhecimento Facial

## Estrutura do Projeto

- `captura_imagens/`
  - `captura_imagens_camera.py`: Código para capturar imagens da câmera.

- `treinamento/`
  - `treino_modelo.py`: Código para treinar o modelo com as imagens capturadas.
  - `imagens/`
    - `captured_faces/`: Imagens capturadas para treinamento.
    - `treino/`: Imagens de treino.
    - `teste/`: Imagens de teste.
  - `modelos/`
    - `modelo_lbph.yml`: Modelo treinado salvo.

- `predicao/`
  - `predicao_camera.py`: Código para realizar a predição em tempo real usando a câmera.

- `utils/`
  - `utils.py`: Funções utilitárias como padronizar_imagem.

## Instruções

### 1. Captura de Imagens
Execute `captura_imagens/captura_imagens_camera.py` para capturar imagens da câmera e salvá-las no diretório `treinamento/imagens/captured_faces/`.

### 2. Treinamento do Modelo
Execute `treinamento/treino_modelo.py` para treinar o modelo com as imagens capturadas e salvar o modelo treinado em `treinamento/modelos/modelo_lbph.yml`.

### 3. Pred
