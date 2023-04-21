import os
import zipfile
import shutil

# Altere isso para o caminho onde estão seus arquivos compactados
diretorio_base = 'D:/luizg/Documents/dataSets/pasta KITTI/KITTI/data_raw/'

# Nome da pasta que você deseja extrair
pasta_desejada = 'image_02'

for item in os.listdir(diretorio_base):
    caminho_item = os.path.join(diretorio_base, item)

    # Verifique se o item é um arquivo ZIP
    if os.path.isfile(caminho_item) and item.endswith('.zip'):
        # Crie uma pasta com o mesmo nome do arquivo ZIP (sem a extensão)
        pasta_destino = os.path.splitext(caminho_item)[0]
        os.makedirs(pasta_destino, exist_ok=True)

        with zipfile.ZipFile(caminho_item, 'r') as arquivo_zip:
            # Lista todos os arquivos/pastas dentro do arquivo ZIP
            for membro in arquivo_zip.namelist():
                # Verifique se o membro pertence à pasta desejada (image_02)
                if pasta_desejada in membro:
                    # Extraia o membro para a pasta de destino
                    arquivo_zip.extract(membro, pasta_destino)
                    print(f'{membro} extraído com sucesso.')

        # Remova o arquivo ZIP após extrair a pasta desejada
        os.remove(caminho_item)
        print(f'Arquivo {caminho_item} removido com sucesso.')
