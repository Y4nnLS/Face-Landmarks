# Separar imagens em um dicionário por prefixo
faces_dict = {}
for arq in lista_faces_captured:
    partes = arq.split('_')
    if len(partes) >= 2:
        prefixo = partes[0]
        if prefixo not in faces_dict:
            faces_dict[prefixo] = []
        faces_dict[prefixo].append(arq)