from utils import CreateDatasetICVLChallenge
from .BaseModel import BDNN
import copy
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
from . import BasicNetworks

#-----------------------------------------------------------------------------------------------------------------------
## @brief The base model including the necessary data loader for the 2018 CVPR Challenge on spectral super-resolution
#
#-----------------------------------------------------------------------------------------------------------------------
class BaseModel_ICVLChallenge(BDNN):
    def __init__(self, track='RealWorld'):
        super(BaseModel_ICVLChallenge, self).__init__()
        self._track = track

    def name(self):
        return 'BaseModel_ICVLChallenge'

    def _create_data_loader(self):
        patchSize = self._patchSize
        dataset = CreateDatasetICVLChallenge(0, patchSize, self._track)
        # dataset.SetSingleChannelOnly(10)
        datasetTrain = copy.deepcopy(dataset)
        datasetValidate = copy.deepcopy(dataset)
        self.datasetTest = copy.deepcopy(dataset)

        if self._enabSave:
            dataset.SaveConfig(os.path.join(self._mainPath, "dataConfig"))

        # load the training and validation set into memory
        if datasetTrain.InitializeSet('train') != 1:
            return 1
        if datasetValidate.InitializeSet('validation') != 1:
            return 1

        self._dataLoaderTrain = DataLoader(dataset=datasetTrain, shuffle='True', batch_size=self._batchSize)
        self._dataLoaderValid = DataLoader(dataset=datasetValidate, shuffle='True', batch_size=self._batchSize)

    def _get_metadata_impl(self):
        # save network meta data
        modelConfig = {}
        modelConfig['General'] = {}
        modelConfig['General']['Track'] = self._track
        modelConfig['General']['BatchSize'] = self._batchSize
        modelConfig['General']['PatchSize'] = self._patchSize
        modelConfig['General']['Loss'] = self._name_criterion
        modelConfig['Network'] = self._network.get_config()
        return modelConfig

    def _set_metadata_impl(self, config):
        networkInfo = config['Network']

        #
        if 'track' in config['General']:
            self._track = config['General']['track']
        else:
            self._track = 'RealWorld'


        # create the specified network
        name = config['Network']['name']
        print("Used network: {}".format(name))

        if name == 'GenUNetNoPooling':
            self._network = BasicNetworks.GenUNetNoPooling()
        else:
            print("Error, Unkown network type: {}".format(name))

        # configure the network accordingly
        self._network.set_config(networkInfo)

    ## @brief Process the specified input using the current model state
    #
    #  The current model state is used to process the specified data, allFiles. The result is written to the path.
    def execute_test(self, allFiles, path):
        self._set_mode2exec()

        #
        self._patchSize = 256
        print("Performing spectral reconstruction")
        for fileOfChoice, img in allFiles.items():
            # img, curSpecImg, fileOfChoice = self.datasetTest.GetImagePair(indImg)
            print("\t-Current Image: " + fileOfChoice)

            imgHeight = img.shape[0]
            imgWidth = img.shape[1]

            img = np.expand_dims(img, axis=4)
            allRGB = torch.from_numpy(img)
            allRGB = allRGB.permute(3, 2, 0, 1)

            if self._enabCuda:
                allRGB = allRGB.cuda()

            # allocate final reconstruction
            reconstruction = np.empty((imgHeight, imgWidth, 31))

            # ... 1.) reconstruction
            curRow = 0
            curCol = 0
            isLastRow = 0
            isLastCol = 0
            while 1:
                # print ('next row')
                endRow = int(curRow + self._patchSize)
                if endRow > imgHeight:
                    curRow = imgHeight - self._patchSize
                    endRow = imgHeight
                    isLastRow = 1

                while 1:
                    # print('next col')
                    endCol = int(curCol + self._patchSize)
                    if endCol > imgWidth:
                        curCol = imgWidth - self._patchSize
                        endCol = imgWidth
                        isLastCol = 1

                    # if we are at the start of a row or column, ...
                    if curRow == 0:
                        sr_batch = 0
                        sr_pred = 0
                    else:
                        sr_batch = curRow + int(self._patchSize / 4)
                        sr_pred = int(self._patchSize / 4)

                    if curCol == 0:
                        sc_batch = 0
                        sc_pred = 0
                    else:
                        sc_batch = curCol + int(self._patchSize / 4)
                        sc_pred = int(self._patchSize / 4)

                    # if we are at the end
                    if isLastRow:
                        er_batch = endRow
                        er_pred = self._patchSize
                    else:
                        er_batch = curRow + int(self._patchSize * 3 / 4)
                        er_pred = int(self._patchSize * 3 / 4)

                    if isLastCol:
                        ec_batch = endCol
                        ec_pred = self._patchSize
                    else:
                        ec_batch = curCol + int(self._patchSize * 3 / 4)
                        ec_pred = int(self._patchSize * 3 / 4)

                    # run the actual reconstruction
                    data = Variable(allRGB[:, :, curRow:endRow, curCol:endCol]).float()
                    self._input = data
                    self._forward()
                    predTo = self._output.cpu()
                    pred = predTo.data.numpy()
                    pred = np.squeeze(pred)
                    pred = np.transpose(pred, (1, 2, 0))


                    reconstruction[sr_batch:er_batch, sc_batch:ec_batch, :] = \
                        pred[sr_pred:er_pred, sc_pred:ec_pred]

                    if isLastCol:
                        isLastCol = 0
                        curCol = 0
                        break
                    curCol = int (curCol + self._patchSize / 2)


                if isLastRow:
                    break
                else:
                    curRow = int(curRow + self._patchSize / 2)

            np.save(os.path.join(path, fileOfChoice), reconstruction)
        print('Done!\n')