import numpy as np, h5py
from torch.utils.data import Dataset
import torch
import os
import random
import cv2
from SpectralImage import MSpecImage

"""
This function defines the data_loader

Args:
    file_name:  path to the utilized data
"""

class RGB2SpectralDataset(Dataset):
    def __init__(self, folder):
        super(RGB2SpectralDataset, self).__init__()

        self._patchsize = 64
        self._channel = -1
        self._folder = folder
        self.dirT1Spec = os.path.join(folder, 'Train1_Spectral')
        self.dirT2Spec = os.path.join(folder, 'Train2_Spectral')
        self.dirT1RGB = []
        self.dirT1RGB = []
        self._challType = ""
        # In the supplied folder, check for training set 1 and 2, respectively
        self.set1 = self.ReadAllFilesInDir(self.dirT1Spec)
        self.set2 = self.ReadAllFilesInDir(self.dirT2Spec)

        self.patchtype = "determ" # random, determ, all
        self.enabDebug = 0
        self.cntFilesAvailable = 0
        self._useSecondSetOnly = 0
    ## Splits the ICVL dataset into a training, validation and test set according to the supplied percentages
    #   The split is performed on an image level.
    #
    # @percTraining size of the training set in percent, i.e. 0-100
    # @percValidation size of the validation set in percent, i.e. 0-100
    # @percTest size of the test set in percent, i.e. 0-100
    def PerformDataSplit(self, percTraining, percValidation, percTest):

        if percTraining+percValidation+percTest != 100:
            print("Sum of all percentages needs to equal 100")
            return 0

        # split into training and test set
        self.percTraining = percTraining/100    # relative amount of images per set to be placed inside the training set
        self.percTest = percTest/100            # relative amount of image completely ignored by the optimization
        self.percValidation = percValidation/100

        self.cntFiles1 = len(self.set1)
        self.cntFiles2 = len(self.set2)
        cntTraining1 = int(self.percTraining * self.cntFiles1)
        cntTraining2 = int(self.percTraining * self.cntFiles2)
        cntValidation1 = int(self.percValidation * self.cntFiles1)
        cntValidation2 = int(self.percValidation * self.cntFiles2)

        # extract the corresponding indices
        # note that the actual files need be explicitly loaded ...
        self.idces1Train = random.sample(range(0, self.cntFiles1), cntTraining1)
        self.idces2Train = random.sample(range(0, self.cntFiles2), cntTraining2)

        # split the remaining images into test set and validation set
        remIdces1 = [i for i in range(self.cntFiles1) if i not in self.idces1Train]
        remIdces2 = [i for i in range(self.cntFiles2) if i not in self.idces2Train]

        print(len(remIdces1))
        tempIdc1 = random.sample(range(0, len(remIdces1)), cntValidation1)
        tempIdc2 = random.sample(range(0, len(remIdces2)), cntValidation2)

        # the resulting validation set
        self.idces1Val = [remIdces1[i] for i in tempIdc1]
        self.idces2Val = [remIdces2[i] for i in tempIdc2]

        # the resulting test set
        self.idces1Test = [remIdces1[i] for i in range(len(remIdces1)) if i not in tempIdc1]
        self.idces2Test = [remIdces2[i] for i in range(len(remIdces2)) if i not in tempIdc2]

        print("Succesfully parsed ICVL dataset!")
        print("Total amount of images: {}".format(self.cntFiles1 + self.cntFiles2))
        print("Size of training set: {}".format(cntTraining1 + cntTraining2))
        print("Size of validation set: {}".format(cntValidation1 + cntValidation2))
        print("Size of test set: {}".format(len(self.idces1Test) + len(self.idces2Test)))

        print("\n-ICVL dataset 1")
        print("\tTotal amount of images: {}".format(self.cntFiles1))
        print("\tSize of training set: {}".format(cntTraining1))
        print("\tSize of validation set: {}".format(cntValidation1))
        print("\tSize of test set: {}".format(len(self.idces1Test)))

        print("\n-ICVL dataset 2")
        print("\tTotal amount of images: {}".format(self.cntFiles2))
        print("\tSize of training set: {}".format(cntTraining2))
        print("\tSize of validation set: {}".format(cntValidation2))
        print("\tSize of test set: {}".format(len(self.idces2Test)))

        # ---- Only Take the first image for testing purposes -----
        if self.enabDebug:
            cntPerSet = 1
            self.idces1Train = self.idces1Train[0:cntPerSet]
            self.idces2Train = self.idces2Train[0:cntPerSet]
            self.idces1Val = self.idces1Val[0:cntPerSet]
            self.idces2Val = self.idces2Val[0:cntPerSet]
        # --------------------------------------------------------

        return 1

    def SetDebugModeOnOff(self, onOff):
        self.enabDebug = onOff

    def SetPatchSize(self, patchSize):
        self._patchsize = patchSize

    def GetCntImages(self):
        return self.cntFilesAvailable

    def GetImagePair(self, id):
        if (id < 0) or (id > self.cntFilesAvailable):
            print("id outside valid index range")
            return 0

        return self.allRGB[id], self.mspecs[id], self.allNames[id]

    ## Saves the current configuration of the database, e.g. data split in form of indices, not the images themselves
    #
    # @param filename have a guess...
    def SaveConfig(self, filename):
        np.savez(filename, i1val=self.idces1Val, i2val=self.idces2Val, i1train=self.idces1Train,
                 i2train=self.idces2Train, i1test=self.idces1Test, i2test=self.idces2Test, percTrain=self.percTraining,
                 percVal=self.percValidation, percTest=self.percTest, files1=self.set1, files2=self.set2,
                 dir1spec=self.dirT1Spec, dir2spec=self.dirT2Spec, dir1rgb=self.dirT1RGB, dir2rgb=self.dirT2RGB,
                 chan=self._channel, patchsize=self._patchsize, challType=self._challType, patchType=self.patchtype)

    ## Load a configuration of the database, e.g. data split in form of indices, not the images themselves
    #
    # @param filename have a guess...
    def LoadConfig(self, filename):
        npzfile = np.load(filename)
        self.idces1Val = npzfile['i1val']
        self.idces2Val = npzfile['i2val']
        self.idces1Train = npzfile['i1train']
        self.idces2Train = npzfile['i2train']
        self.idces1Test = npzfile['i1test']
        self.idces2Test = npzfile['i2test']
        self.percTraining = npzfile['percTrain']
        self.percTest = npzfile['percTest']
        self.percValidation = npzfile['percVal']
        self.dirT1Spec = str(npzfile['dir1spec'])
        self.dirT2Spec = str(npzfile['dir2spec'])
        self.dirT1RGB = str(npzfile['dir1rgb'])
        self.dirT2RGB = str(npzfile['dir2rgb'])
        self.set1 = npzfile['files1']
        self.set2 = npzfile['files2']
        self._channel = npzfile['chan']
        self._patchsize = npzfile['patchsize']
        print(self._patchsize)

        if 'challType' in npzfile:
            self._challType = npzfile['challType']
            print (self._challType)
        else:
            print('Warning: Old version, manual specification of the challenge type is required!')

    def ReadAllFilesInDir(self, folder):
        allFiles = []
        for file in os.listdir(folder):
            if file.endswith(".mat"):
                allFiles.append(file[:-4])
        return allFiles

    def SetChallengeType(self, type='RealWorld'):
        if type == 'RealWorld':
            self.dirT1RGB = os.path.join(self._folder, 'Train1_RealWorld')
            self.dirT2RGB = os.path.join(self._folder, 'Train2_RealWorld')
        elif type == 'Clean':
            self.dirT1RGB = os.path.join(self._folder, 'Train1_Clean')
            self.dirT2RGB = os.path.join(self._folder, 'Train2_Clean')
        else:
            return 0

        self._challType = type
        return 1


    def InitializeSet(self, type='train'):
        print('Loading images into memory...')

        if self._challType == "":
            print("Error! Challenge type has not been specified.")
            return 0

            # ---- Only Take the first image for testing purposes -----
        if self.enabDebug:
            cntPerSet = 1
            self.idces1Train = self.idces1Train[0:cntPerSet]
            self.idces2Train = self.idces2Train[0:cntPerSet]
            self.idces1Val = self.idces1Val[0:cntPerSet]
            self.idces2Val = self.idces2Val[0:cntPerSet]
        # --------------------------------------------------------

        idces1 = []
        idces2 = []
        if type == 'train':
            idces1 = self.idces1Train
            idces2 = self.idces2Train
        elif type == 'validation':
            idces1 = self.idces1Val
            idces2 = self.idces2Val
        elif type == 'test':
            idces1 = self.idces1Test
            idces2 = self.idces2Test
        else:
            return 0

        # idces1 = idces1[0:1]
        # idces2 = idces2[0:1]

        # _______________________________________________________________________________________________
        #
        # ... 1 load all files
        self.cntFilesAvailable = len(idces1) + len(idces2)
        self.mspecs = []
        self.allRGB = []
        self.allNames = []

        #  load every indexed file within the first set
        if self._useSecondSetOnly == 0:
            for c, indFile in enumerate(idces1):
                curName = self.set1[indFile]

                # load the spectral image, i.e. the ground truth
                filename_spectral = os.path.join(self.dirT1Spec, curName + '.mat')
                print(filename_spectral)
                curSpecImg = MSpecImage()
                curSpecImg.LoadICVLSpectral(filename_spectral)
                self.mspecs.append(curSpecImg)
                self.allNames.append(curName)

                # load the rgb image
                if self._challType == "Clean":
                    filename_rgb = os.path.join(self.dirT1RGB, curName + '_clean.png')
                elif self._challType == "RealWorld":
                    filename_rgb = os.path.join(self.dirT1RGB, curName + '_camera.jpg')
                self.allRGB.append(cv2.imread(filename_rgb))

        print("first set loaded")
        #  load every indexed file within the second set
        for indFile in idces2:
            self.cntFilesAvailable = len(idces2)
            curName = self.set2[indFile]

            # load the spectral image, i.e. the ground truth
            filename_spectral = os.path.join(self.dirT2Spec, curName + '.mat')
            print(filename_spectral)
            curSpecImg = MSpecImage()
            curSpecImg.LoadICVLSpectral(filename_spectral)
            self.mspecs.append(curSpecImg)
            self.allNames.append(curName)

            # load the rgb image
            if self._challType == "Clean":
                filename_rgb = os.path.join(self.dirT2RGB, curName + '_clean.png')
            elif self._challType == "RealWorld":
                filename_rgb = os.path.join(self.dirT2RGB, curName + '_camera.jpg')
            self.allRGB.append(cv2.imread(filename_rgb))

        # _______________________________________________________________________________________________
        #
        # 2) Convert loaded files into patches
        cntTotalPatches = 0
        cntTotalPatchesRand = 0
        for c in range(0, self.cntFilesAvailable):
            if self.patchtype == 'all':
                cntTotalPatches += self.mspecs[c].GetCntPossiblePatchesAll(self._patchsize, self._patchsize)
            elif self.patchtype == 'determ':
                cntTotalPatches += self.mspecs[c].GetCntPossiblePatches(self._patchsize, self._patchsize)
            elif self.patchtype == 'random':
                cntTotalPatches += self.mspecs[c].GetCntPossiblePatches(self._patchsize, self._patchsize)
                cntTotalPatchesRand += self.mspecs[c].GetCntPossiblePatchesAll(self._patchsize, self._patchsize)

        print("Total amount of available patches: {}".format(cntTotalPatches))
        self.idxMap = [-1] * cntTotalPatches, [-1] * cntTotalPatches
        lastVal = 0
        for c in range(0, self.cntFilesAvailable):
            if self.patchtype == 'all':
                cntCurPatches = self.mspecs[c].GetCntPossiblePatchesAll(self._patchsize, self._patchsize)
            elif self.patchtype == 'determ' or self.patchtype == 'random':
                cntCurPatches = self.mspecs[c].GetCntPossiblePatches(self._patchsize, self._patchsize)

            self.idxMap[0][lastVal:lastVal+cntCurPatches] = [c] * cntCurPatches
            self.idxMap[1][lastVal:lastVal + cntCurPatches] = np.arange(cntCurPatches).tolist()

            lastVal = lastVal+cntCurPatches

        if self.patchtype == 'random':
            print("Random mode active. Total amount underlying of patches: {}".format(cntTotalPatchesRand))
            self.shuffledIdces = list(np.ndindex(len(self.idxMap[0])))
            # Shuffle the indices in-place
            np.random.shuffle(self.shuffledIdces)
            self.nextAccessed = 0
        self.cntDataAvailable = cntTotalPatches

        print('Done!')
        return 1

    def SetSingleChannelOnly(self, channel):
        self._channel = channel

    def __len__(self):
        return self.cntDataAvailable

    def __getitem__(self, item):

        if self.patchtype == 'random':
            indImg = self.idxMap[0][self.shuffledIdces[self.nextAccessed][0]]
            indPatch = self.idxMap[1][self.shuffledIdces[self.nextAccessed][0]]
            self.nextAccessed += 1

            if self.nextAccessed == len(self.shuffledIdces):
                self.nextAccessed = 0
                np.random.shuffle(self.shuffledIdces)
        else:
            # get image pair the item corresponds to
            indImg = self.idxMap[0][item]
            indPatch = self.idxMap[1][item]

        # get the image area corresponding to the patch id
        if self.patchtype == 'all' or self.patchtype == 'random':
            startR, startC = self.mspecs[indImg].GetPatchStartAll(indPatch, self._patchsize, self._patchsize)
        elif self.patchtype == 'determ':
            startR, startC =  self.mspecs[indImg].GetPatchStart(indPatch, self._patchsize, self._patchsize)

        x = self.allRGB[indImg][startR:startR+self._patchsize, startC:startC+self._patchsize]

        midPoint = int(self._patchsize / 2)
        y = self.mspecs[indImg].data[startR:startR+self._patchsize, startC:startC+self._patchsize, :]

        #
        if self._channel >= 0:
            y = y[:,:,self._channel]
            y = torch.from_numpy(y)
            x = torch.from_numpy(x)
            x = x.permute(2, 0, 1) # torch expects the following ordering: (channels, height, width)
        else:
            y = torch.from_numpy(y)
            x = torch.from_numpy(x)
            y = y.permute(2, 0, 1)  # torch expects the following ordering: (channels, height, width)
            x = x.permute(2, 0, 1)  # torch expects the following ordering: (channels, height, width)



        return x, y


    # def CreatePatches(self, img, patchsize, overlap):























class SpectralData(Dataset):

    def __init__(self, folder):
        super(SpectralData, self).__init__()


        self.folder = folder
        self.allFiles = []
        for file in os.listdir(folder):
            if file.endswith(".mat"):
                self.allFiles.append(file)
        self.cntFiles = len(self.allFiles)


        # load data
        # self.spectra = np.load('/home/staff/stiebel/Code/Python/mspec/SpectralReconstruction/test.npy')

    def InitializeSet(self, type='train'):
        cnter = 0
        cntFilesTrain = 1
        cntFilesValidation = 1
        cntSpectra = 1300 * 1392

        allSpecs = []

        print('Initializing set...')

        if type == 'train':
            # simply load the first 4 files
            for indFile in range(0, cntFilesTrain):
                filename = os.path.join(self.folder, self.allFiles[indFile])
                f = h5py.File(filename)
                curRs = np.array(f['rad'])
                curRs = curRs.reshape(31, cntSpectra)
                allSpecs.append(curRs.transpose())

        elif type == 'validation':
            # simply load the first 4 files
            for indFile in range(cntFilesTrain, cntFilesTrain+cntFilesValidation):
                filename = os.path.join(self.folder, self.allFiles[indFile])
                f = h5py.File(filename)
                curRs = np.array(f['rad'])
                curRs = curRs.reshape(31, cntSpectra)
                allSpecs.append(curRs.transpose())

        self.spectra = np.vstack(allSpecs)

        print('Done!')

    # mandatory function for number of samples
    def __len__(self):
        print(len(self.spectra))
        return len(self.spectra)

    # mandatory function to get a specific sample
    def __getitem__(self, item):
        curSpec = self.spectra[item, :]
        x = torch.from_numpy(curSpec)
        y = x

        return x, y

class FullData(Dataset):

    def __init__(self, file_name):
        super(FullData, self).__init__()

        # Assuming the file is a hdf5 file with 3 matrices: input, label and mask
        # In addition, all subjects (in this example 10) are stored as channel (4th dimension)
        self.file_name = file_name
        f_data = h5py.File(self.file_name)
        self.full_data = np.transpose(np.array(f_data['/input']))
        self.full_label = np.transpose(np.array(f_data['/label']))
        self.full_mask = np.transpose(np.array(f_data['/mask']))
        print('complete dataset loaded...')
        f_data.close()

    # This function returns the corresponding dataset
    def set_type(self, set_type='validation'):
        if set_type == 'train':
            print('Loading train dataset')
            x = self.full_data
            x = x[:, :, :, np.arange(8)]
            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
            self.x = torch.from_numpy(x)

            label = self.full_label
            label = label[:, :, :, np.arange(8)]
            label = np.reshape(label, (label.shape[0], label.shape[1], label.shape[2]*label.shape[3]))
            self.label = torch.from_numpy(label)

            mask = self.full_mask
            mask = mask[:, :, :, np.arange(8)]
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], mask.shape[2]*mask.shape[3]))
            self.indices = np.where(mask > 0)

            self.data = self.x, self.label, self.indices

        elif set_type == 'validation':
            print('Loading validation dataset')

            x = self.full_data
            x = x[:, :, :, 8]
            x = np.squeeze(x)
            self.x = torch.from_numpy(x)

            label = self.full_label
            label = label[:, :, :, 8]
            label = np.squeeze(label)
            self.label = torch.from_numpy(label)

            mask = self.full_mask
            mask = mask[:, :, :, 8]
            mask = np.squeeze(mask)
            self.indices = np.where(mask > 0)

            self.data = self.x, self.label, self.indices

        else:
            print('Loading test dataset')

            x = self.full_data
            x = x[:, :, :, 9]
            x = np.squeeze(x)
            self.x = torch.from_numpy(x)

            label = self.full_label
            label = label[:, :, :, 9]
            label = np.squeeze(label)
            self.label = torch.from_numpy(label)

            mask = self.full_mask
            mask = mask[:, :, :, 9]
            mask = np.squeeze(mask)
            self.indices = np.where(mask > 0)

            self.data = self.x, self.label, self.indices

    # mandatory function for number of samples
    def __len__(self):
        return len(self.data[2][0])

    # mandatory function to get a specific sample
    def __getitem__(self, item):
        idx = self.data[2][0][item]
        idy = self.data[2][1][item]
        idz = self.data[2][2][item]

        x = self.data[0][idx - 2:idx + 3, idy - 2:idy + 3, idz - 2:idz + 3]
        x = torch.unsqueeze(x, 0)
        y = self.data[1][idx - 2:idx + 3, idy - 2:idy + 3, idz - 2:idz + 3]
        y = torch.unsqueeze(y, 0)

        return x, y

