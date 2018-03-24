#!/usr/bin/env python3

from networks.Models import BaseModel_ICVLChallenge
from SpectralImage import MSpecImage
import os
from utils import ComputeErrorMetrics
import numpy as np
import cv2
import torch

#
if torch.cuda.is_available():
    print("Cuda is available")
else:
    print("Cuda is not available, using CPU only")

# configuration
perform_rec = 1

epoch = 10

# set paths
path2load = ""                  # folder of the model to load
folderRekonstr = path2load + "rekonstr\\"
path_rec = folderRekonstr       # folder to store the computed reconstruction in
imageFolder = ""                # folder containing all images to be loaded
path_gt = ""                    # folder containing the corresponding ground truth


# ...   load the rgb images
allImgs = {}
for file in os.listdir(imageFolder):
    if file.endswith(".jpg"):
        filename_rgb = os.path.join(imageFolder, file)
        allImgs[file[:-11]] = cv2.imread(filename_rgb)
    if file.endswith(".png"):
        filename_rgb = os.path.join(imageFolder, file)
        allImgs[file[:-10]] = cv2.imread(filename_rgb)

# ...  perform the reconstruction
if perform_rec:
    # create the model
    model = BaseModel_ICVLChallenge()
    model.load_metadata(path2load)
    model.load_model_state(path2load, epoch)

    # create reconstruction folder
    if not os.path.exists(folderRekonstr):
        os.makedirs(folderRekonstr)

    # perform the reconstruction
    model.ExecuteTest(allImgs, folderRekonstr)

# ...   load the spectral images
err_file = open(folderRekonstr + 'err.txt', 'w')

err_file.write('Results:\n')

MRAEs = {}
RMSEs = {}
print('Reading validation data set')
for file in os.listdir(path_gt):
    if file.endswith(".mat"):
        curName = file[:-4]
        print(curName)

        # load gt
        spec_gt = MSpecImage()
        spec_gt.LoadICVLSpectral(os.path.join(path_gt, file))

        # load est
        spec_rec = MSpecImage()
        curImg = np.load(os.path.join(path_rec, curName+".npy"))
        spec_rec.data = curImg

        # calculate error
        mrae, rmse = ComputeErrorMetrics(spec_gt.data, spec_rec.data)
        MRAEs[curName] = mrae
        RMSEs[curName] = rmse

        err_file.write(curName + ': MRAE {} - RMSE {}\n'.format(MRAEs[curName], RMSEs[curName]))
        print(MRAEs[curName])
        print(RMSEs[curName])

print('Evaluating...')

# run the evaluation as supplied
MRAEs_values = [MRAEs[key] for key in MRAEs]
RMSEs_values = [RMSEs[key] for key in RMSEs]

MRAE = np.mean(MRAEs_values)
print("MRAE:\n" + MRAE.astype(str))
RMSE = np.mean(RMSEs_values)
print("\nRMSE:\n" + RMSE.astype(str))

err_file.write("\nFinal Results:\n")
err_file.write(curName + ': MRAE {} - RMSE {}'.format(MRAE,RMSE))

err_file.close()
