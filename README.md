# Projeto de Reconhecimento Facial

Este projeto realiza o reconhecimento facial utilizando três diferentes algoritmos: LBPH (Local Binary Patterns Histograms), Fisherface, e Eigenface. O projeto captura imagens de rostos, treina modelos com essas imagens e permite a predição em tempo real utilizando os modelos treinados.

## Estrutura do Projeto

```
.
├── main.py
├── capture_image
|   └── capture_images_camera.py
├── predict
|   └── prediction.py
├── training
|   ├── training_model.py
│   ├── images
│   │   ├── captured_faces
│   │   └── training
│   └── models
│       ├── model_lbph.yml
│       ├── model_fisherface.yml
│       ├── model_eigenface.yml
│       └── subject_map.json
└── README.md
```

## Requisitos

Certifique-se de ter o Python instalado. Instale as bibliotecas necessárias utilizando o `pip`:

```bash
pip install opencv-python
```

```bash
pip install mediapipe
```

```bash
pip install scikit-learn
```

```bash
pip install termcolor
```

## Como Funciona

### 1. Captura de Imagens (`capture_images_camera.py`)

Este script captura imagens da webcam e salva em um diretório para treinamento.

- Solicita o nome do sujeito.
- Captura 50 imagens do rosto do sujeito.
- Salva as imagens no diretório `./training/images/captured_faces`.

Para executar o script:

```bash
python capture_images_camera.py
```

### 2. Treinamento do Modelo (`training_model.py`)

Este script treina três diferentes modelos de reconhecimento facial com as imagens capturadas.

- Organiza as imagens no diretório `./training/images/training`.
- Aplica normalização e aumento de dados nas imagens.
- Divide os dados em conjuntos de treinamento e teste.
- Treina os modelos LBPH, Fisherface e Eigenface.
- Avalia a precisão dos modelos.
- Salva os modelos treinados no diretório `./training/models`.

Para executar o script:

```bash
python training_model.py
```

### 3. Predição em Tempo Real (`prediction.py`)

Este script utiliza a webcam para capturar imagens em tempo real e realiza a predição utilizando os modelos treinados.

- Seleciona o modelo de predição.
- Captura imagens da webcam.
- Realiza a predição com o modelo selecionado.
- Exibe o nome do sujeito identificado na imagem.

Para executar o script:

```bash
python prediction.py
```

### 4. Interface do Menu (`main.py`)

Este script fornece uma interface de menu para selecionar o modelo de predição e realizar a predição em tempo real.

- Apresenta um menu com opções para LBPH, Fisherface, Eigenface e sair.
- Executa a predição com o modelo selecionado.

Para executar o script:

```bash
python main.py
```

## Detalhes do Código

### `capture_images_camera.py`

- Utiliza OpenCV para capturar imagens da webcam.
- Utiliza Mediapipe para detectar rostos.
- Salva imagens de rostos detectados em `./training/images/captured_faces`.

### `training_model.py`

- Lê as imagens capturadas e organiza para treinamento.
- Aplica normalização e aumento de dados.
- Utiliza `train_test_split` do Scikit-learn para dividir os dados.
- Treina três modelos de reconhecimento facial (LBPH, Fisherface, Eigenface).
- Avalia e imprime a precisão dos modelos.
- Salva os modelos treinados e o mapeamento de sujeitos.

### `prediction.py`

- Lê o modelo selecionado e o mapeamento de sujeitos.
- Utiliza Mediapipe para detectar rostos em imagens capturadas da webcam.
- Realiza a predição e exibe o resultado na imagem capturada.

### `main.py`

- Exibe um menu para selecionar o modelo de predição.
- Executa o script de predição com o modelo selecionado.
