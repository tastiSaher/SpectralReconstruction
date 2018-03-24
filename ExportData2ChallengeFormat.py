import os
import numpy as np
import h5py

imageFolder = "K:\\rgb2spec\\RealWorld\\Unet\\Rekonstr"
for file in os.listdir(imageFolder):
    if file.endswith(".npy"):
        filename = os.path.join(imageFolder, file)
        curImg = np.load(filename)
        curImg = np.swapaxes(curImg, 0, 2)

        # write file
        f = h5py.File(os.path.join(imageFolder, file[:-4])+".mat", "w")
        dset = f.create_dataset('rad', data=curImg)
        f.close()