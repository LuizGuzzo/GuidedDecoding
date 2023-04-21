import numpy as np
from PIL import Image
from fill_depth_colorization import fill_depth_colorization

half = (640,192)

# data = np.load('D:/luizg/Documents/dataSets/KITTI_test/2011_09_26_2011_09_26_drive_0002_sync_02_data_0000000000.npz')

path = "D:/luizg/Documents/dataSets/kitti/depth_selection/test_depth_completion_anonymous"
name = "0000000027"

# aplicando fill
image = Image.open(path+"/image/"+name+".png")
sparseDepth = Image.open(path+"/velodyne_raw/"+name+".png")
image = np.array(image)
sparseDepth = np.array(sparseDepth)
denseDepth = fill_depth_colorization(image,sparseDepth)

# save test
denseDepth = np.interp(denseDepth, (denseDepth.min(), denseDepth.max()), (0, 255))
denseDepth = Image.fromarray(np.uint8(denseDepth))
denseDepth.save(path+"/denseDepth/"+name+".png")

# np.savez(path+"/results/dados.npz", image=image, depth=denseDepth)


# load test
# dados = np.load(path+"/results/"+name+".npz")
# imagem = dados['image']
# profundidade = dados['depth']


print("para")