import os
import shutil

# Altere isso para o caminho onde estão suas 28 pastas principais
diretorio_base = 'D:\luizg\Documents\dataSets\pasta_KITTI\KITTI\data_depth_annotated'

# Lista de pastas para remover
pastas_para_remover = ['image_00', 'image_01', 'image_03', 'velodyne_points', 'oxts']

def remover_pastas_indesejadas(caminho_atual):
    for pasta in pastas_para_remover:
        caminho_pasta_para_remover = os.path.join(caminho_atual, pasta)

        # Verifique se a pasta existe antes de tentar removê-la
        if os.path.exists(caminho_pasta_para_remover):
            shutil.rmtree(caminho_pasta_para_remover)
            print(f'Pasta {caminho_pasta_para_remover} removida com sucesso.')

    for item in os.listdir(caminho_atual):
        caminho_item = os.path.join(caminho_atual, item)

        # Verifique se o caminho é uma pasta (e não um arquivo)
        if os.path.isdir(caminho_item):
            remover_pastas_indesejadas(caminho_item)

remover_pastas_indesejadas(diretorio_base)
