from MulticoreTSNE import MulticoreTSNE as TSNE
from time import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torchfile
import h5py as h5
from PIL import Image
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def getFeatures(inputFile, h5Matrix):
    h5f = h5.File(inputFile, "r")
    feat = h5f[h5Matrix][:]
    h5f.close()
    return feat

# dataset = torchfile.load(sys.argv[1])
# data = getFeatures(sys.argv[2], "features")
# perplexity = int(sys.argv[3])
# angle = float(sys.argv[4])
# n_iter = int(sys.argv[5])
# save_name = sys.argv[6]


dataset = "/home/mlagunas/Bproject/DLart/data/paths/curated/curated_paths_test.txt"
data = getFeatures(
    "/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196/h5/curated/features/vgg19/curated_vgg19_test_42.h5", "features")
perplexity = 150
angle = 0.5
n_iter = 1500
save_name = ""

paths = [line.rstrip('\n') for line in open(dataset)]


tsne = TSNE(n_jobs=8, perplexity=perplexity, angle=angle, n_iter=n_iter)
data = data.astype(np.float64)
vis_data = tsne.fit_transform(data)
x_data = vis_data[:,[0]]
y_data = vis_data[:,[1]]

fig = plt.gcf()
fig.clf()
fig.set_size_inches(18.5, 10.5, forward=True)
ax = plt.subplot(111)
ax.set_axis_off()
size = 256, 256
# add a first image
for i in range(0, len(paths)):
    img = Image.open(paths[i])
    img.thumbnail(size, Image.ANTIALIAS)
    ab = AnnotationBbox(OffsetImage(img, zoom=.15, cmap='gray'),
                        [x_data[i], y_data[i]],
                        frameon=False,
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points")

    ax.add_artist(ab)
    ax.plot(x_data[i], y_data[i], 'k.', markersize=2)
# rest is just standard matplotlib boilerplate
# ax.grid(True)
# plt.title('t-SNE using the embeddings obtained with the optimized deep neural network')
plt.xlim(x_data.min(), x_data.max())
plt.ylim(y_data.min(), y_data.max())
plt.xticks(())
plt.yticks(())
# plt.draw()
# plt.show()
# fig.set_dpi(4800)
fig.savefig("../../data/tsne.svg",transparent=True,dpi = 300, bbox_inches='tight', pad_inches=1)
