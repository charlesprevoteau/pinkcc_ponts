import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random as rd 
import matplotlib.pyplot as plt
import os


"""
## Visualisation pour une image : 

i_coupe = rd.randint(0,129)   # slice 34 pour voir zone jaune
print("i_coupe : ", i_coupe)

## Chargement des images

image_CT_path = 'data/dataset_npy/CT/330680_CT.npy'
image_segm_path = 'data/dataset_npy/Segmentation/330680_segm.npy'

image_CT_data = np.load(image_CT_path)
image_segm_data = np.load(image_segm_path)

# Affichage de l'image 

image_CT = image_CT_data[:,:, 34]
plt.subplot(1,2,1)
plt.imshow(image_CT, cmap='grey')

image_segm = image_segm_data[:,:,34]
plt.subplot(1,2,2)
plt.imshow(image_segm)

plt.show()
"""

repertoire = 'data/dataset_npy/CT'
liste_image = os.listdir(repertoire)
l = rd.randint(0, 129)

for i in range(9):

    j = rd.randint(0,len(liste_image)-1)
    image = np.load(repertoire + '/' + liste_image[j])
    coupe_image = image[:,:,l]
    plt.subplot(3,3, i+1)
    plt.imshow(coupe_image, cmap='grey')

plt.show()


