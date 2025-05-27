"""
A utiliser pour traiter le dataset. 

traitement(ct_path) : Charger chaque image, normaliser pour chaque image les valeurs de gris
 + stocker image segmentée sous forme d'entiers (moins de mémoire). Enregistre les infos dans
des fichiers .npy, pour éviter d'avoir à tout retraiter avant chaque entrainement. 

split_dataset(seed_value (None si aléatoire/réel si on veut reproduire une distribution
particulière)) : Scinde en 2 la base de données selon la répartition : ratio_train dans train 
et (1-ratio_train) dans test. 
"""

import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from pathlib import Path
import random as rd
import pandas
from shutil import copyfile

ct_path = Path('data/dataset/CT')

def traitement(ct_path=ct_path):

    liste_paths = list(ct_path.glob('MSKCC/*.nii.gz')) + list(ct_path.glob('TCGA/*.nii.gz'))

    for image_CT_path in liste_paths:

        scaler = MinMaxScaler()

        origine_name = image_CT_path.parts[-2]     # donne TCGA ou MSKCC
        nom_image = f"{image_CT_path.stem.replace('.nii', '')}"  # donne seulement le nom de l'image
        image_segm_path = Path(f'data/dataset/Segmentation/{origine_name}/{nom_image}.nii.gz')
        
        print(f"Traitement de l'image {nom_image}")

        # Chemins de sauvegarde pour les fichiers .npy
        save_ct_path = f'data/dataset_npy/CT/{nom_image}_CT.npy'
        save_segm_path = f'data/dataset_npy/Segmentation/{nom_image}_segm.npy'
        if os.path.exists(save_ct_path) and os.path.exists(save_segm_path):
            print(f"L'image {nom_image} a déjà été traitée. Passage à l'image suivante.")
            continue

        image_CT_data = nib.load(image_CT_path).get_fdata()
        image_CT_2D = image_CT_data.reshape(-1, image_CT_data.shape[-1])
        normalised_image_CT_2D = scaler.fit_transform(image_CT_2D)
        normalised_image_CT_3D = normalised_image_CT_2D.reshape(image_CT_data.shape)

        image_segm_data = nib.load(image_segm_path).get_fdata()
        image_segm_data = image_segm_data.astype(np.uint8)

        np.save(save_ct_path, normalised_image_CT_3D)
        np.save(save_segm_path, image_segm_data)

        print(f"L'image {nom_image} a été traitée")

    print("Le traitement du dataset est terminé.")



def split_dataset(seed_value, ratio_train, data_folder_npy='data/dataset_npy'):

    n = 1

    # Création d'un nouvel ensemble de données train/test
    for dossier in os.listdir('data'):
        if 'split' in dossier : 
            n+=1
    
    output_dataset = f'data/dataset_npy_split_{n}'
    os.makedirs(output_dataset, exist_ok=True)

    data_folder_CT = data_folder_npy + '/CT/'
    data_folder_segm = data_folder_npy + '/Segmentation/'

    train_path = output_dataset + '/train'
    os.makedirs(train_path, exist_ok=True)
    test_path = output_dataset + '/test'
    os.makedirs(test_path, exist_ok=True)

    # Si seed_value vaut None, tirage aléatoire, sinon, on peut reproduire le tirage aléatoire dont seed vaut seed_value 
    if seed_value is not None:
        rd.seed(seed_value)

    train_CT_path = train_path + '/CT/'
    train_segm_path = train_path + '/Segmentation/'
    test_CT_path = test_path + '/CT/'
    test_segm_path = test_path + '/Segmentation/'

    os.makedirs(train_CT_path, exist_ok=True)
    os.makedirs(train_segm_path, exist_ok=True)
    os.makedirs(test_CT_path, exist_ok=True)
    os.makedirs(test_segm_path, exist_ok=True)

    for image_CT in os.listdir(data_folder_CT):

        image_CT_path = data_folder_CT + image_CT
        image_segm = image_CT.replace("CT.npy","segm.npy")
        image_segm_path = data_folder_segm + image_segm

        if rd.random() < ratio_train : 
            copyfile(image_CT_path, train_CT_path + image_CT)
            copyfile(image_segm_path, train_segm_path + image_segm)
            print(f"L'image {image_CT_path} et la version _segm sont des données d'entrainement")

        else:
            copyfile(image_CT_path, test_CT_path + image_CT)
            copyfile(image_segm_path, test_segm_path + image_segm)
            print(f"L'image {image_CT_path} et la version _segm sont des données de test")

    print('Toutes les données ont bien été réparties dans 2 dossier /train et /test')

#traitement()
#split_dataset(None, 0.8)


