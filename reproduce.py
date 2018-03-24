#!/usr/bin/env python3

import os, sys
import torch
import imageio
import configparser
import numpy as np
import h5py
from networks.Models import BaseModel_ICVLChallenge

if __name__ == '__main__':
    print("Preparing to predict spectral images...")

    # check for cuda support
    if torch.cuda.is_available():
        print("Cuda is available!")
    else:
        print("Cuda is not available, using CPU only!")


    # ...   open config file

    # check if file exists
    configname = 'config.ini'
    config = configparser.ConfigParser()
    if os.path.exists(configname):
        print('\nReading config file...')
    else:
        sys.exit('The configuration file {} is missing. Please refer to the readme for detailed '
                 'information.'.format(configname))

    # load file
    config.read(configname)
    track = config['Info']['track']
    path2images = config['Info']['path2images']
    path2store = config['Info']['path2store']

    # configure accordingly
    if track == 'Clean':
        path2load = './model_clean/'
    elif track == 'RealWorld':
        path2load = './model_real/'
    else:
        sys.exit('\tThe specified track {} is not recognized. Valid options are either RealWorld or Clean.'.format(track))
    print('\tconfigured to load the network trained on {} data'.format(track))


    # ...   search for rgb images in the specified directory
    print('\tsearching for image files in the specified directory...'.format(track))
    allImgs = {}
    allImageNames = []
    found_jpg = 0
    found_png = 0

    # loop over all files and store all .jpg and .png files
    for file in os.listdir(path2images):
        if file.endswith(".jpg"):
            filename_rgb = os.path.join(path2images, file)
            allImageNames.append(filename_rgb)
            allImgs[file[:-11]] = imageio.imread(filename_rgb)
            found_jpg = 1
        if file.endswith(".png"):
            filename_rgb = os.path.join(path2images, file)
            allImageNames.append(filename_rgb)
            allImgs[file[:-10]] = imageio.imread(filename_rgb)
            found_png = 1

    for name in allImageNames:
        print('\t\t{}'.format(name))

    if len(allImageNames) == 0:
        sys.exit('\t\tFailed to locate any image in the path {}'.format(path2images))

    # perform validity check
    if found_jpg and track == 'Clean':
        print(
            '\tWarning: jpg-files were detected but the track is set to Clean. In the original challenge, jpg-files '
            'originated from the RealWorld-track only. If that is still the case, images from the RealWorld-track will '
            'be processed using the network trained upon images from the Clean-track possibly leading to inferior '
            'results.')
    if found_png and track == 'RealWorld':
        print('\tWarning: png-files were detected but the track is set to RealWorld. In the original challenge, png-files '
              'originated from the Clean-track only. If that is still the case, images from the Clean-track will be '
              'processed using the network trained upon images from the RealWorld-track possibly leading to inferior '
              'results.')

    print('Configuration done!\n')


    print('Creating model...')

    # create the model
    model = BaseModel_ICVLChallenge()
    print('\t-loading meta data...')
    model.load_metadata(path2load)
    print('\t-deserializing...')
    model.load_model_state(path2load, -1)
    print('Model created!\n')

    # create reconstruction folder
    if not os.path.exists(path2store):
        os.makedirs(path2store)

    # perform the reconstruction
    model.execute_test(allImgs, path2store)


    # ...   convert reconstructed files into challenge format
    print('Converting files to challenge format...')

    for file in os.listdir(path2store):
        if file.endswith(".npy"):
            print(file)
            filename = os.path.join(path2store, file)
            curImg = np.load(filename)
            curImg = np.swapaxes(curImg, 0, 2)

            # write file
            f = h5py.File(os.path.join(path2store, file[:-4]) + ".mat", "w")
            dset = f.create_dataset('rad', data=curImg)
            f.close()