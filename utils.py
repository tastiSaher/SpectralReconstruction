import torch
import data
import os
import numpy as np

def SaveNet(path, net, netName, curEpoch):
    filename = os.path.join(path, netName + "_epoch{}.mod".format(curEpoch))
    torch.save(net.state_dict(), filename)

def CreateDatasetICVLChallenge(useCommonDataSplit, patchSize, track):
    imageFolder = "/images/mspec/Bilder_anderer_Gruppen/ICVL/"
    dataset = data.RGB2SpectralDataset(imageFolder)
    dataset.enabDebug = 0
    dataset._useSecondSetOnly = 0
    dataset.patchtype = 'determ'

    if useCommonDataSplit:
        # dataset.LoadConfig("/home/temp/stiebel/rgb2spec/RealWorld/dataConfig30perc.npz")
        dataset.LoadConfig("/home/temp/stiebel/rgb2spec/RealWorld/patchsize64/UNet_Depth5_random/dataConfig.npz")
    else:
        dataset.PerformDataSplit(100, 0, 0)

    dataset.SetChallengeType(track)
    dataset.SetPatchSize(patchSize)
    return dataset

def ComputeErrorMetrics(gt, rc):

    # compute MRAE
    diff = gt-rc
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff,gt+np.finfo(float).eps) # added epsilon to avoid division by zero.

    MRAE = np.mean(relative_abs_diff)

    # compute RMSE
    square_diff = np.power(diff,2)
    RMSE = np.sqrt(np.mean(square_diff))

    return MRAE, RMSE