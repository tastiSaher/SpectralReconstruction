import os
import torch, torch.nn
import torch.optim as optim
from torch.autograd import Variable
from Logging import Log
from collections import OrderedDict
from utils import SaveNet
import configparser
from . import CustomLoss

#-----------------------------------------------------------------------------------------------------------------------
## @brief The most abstract base model
#
#  All subclasses need to implement the functions: Forward, Backward
#   optional: InitializeTraining
#-----------------------------------------------------------------------------------------------------------------------
class BaseModel():
    def __init__(self):
        self._learnRate = 0.0001
        self._cntEpoch = 0
        if torch.cuda.is_available():
            self._enabCuda = 1
        else:
            self._enabCuda = 0

        self._logger = []
        self._enabSave = 0
        self._enab_debug_output = 0

    #___________________________________________________________________________________________________________________
    #
    # Functionality

    ## @brief Returns the model name
    #
    #  @note    This function must be overridden by the subclass
    def name(self):
        return 'BaseModel'

    ## @brief Enables logging under the specified path
    #
    #  The created log file is 'path/log.txt'
    def enab_logging(self, path):
        self._enabSave = 1
        self._mainPath = path
        self._logger = Log(path)

    ## @brief Executes a training pass
    #
    #  ...
    def execute_training(self):

        self._set_mode2train()
        self._trainLoss = 0

        # log the current epoch
        self._logger.DisplayEpoch(self._cntEpoch)
        self._cntEpoch += 1

        for batch_idx, (data, label) in enumerate(self._dataLoaderTrain):
            # get current input data and the corresponding label
            self._input = Variable(data).float()
            self._label = Variable(label).float()
            if self._enabCuda:
                self._input = self._input.cuda()
                self._label = self._label.cuda()

            # optimize according to specified routine
            self._optimize_params()

            # log the training progress
            if batch_idx % 1 == 0:
                self._logger.DisplayErrors(batch_idx, len(self._dataLoaderTrain), self.get_loss())

    ## @brief Executes a validation pass
    #
    #  The current network state is used to process the specified validation data, self._dataLoaderValid, e.g. calculate
    #  a prediction. The processed data is compared against the label and the mean loss, called validation loss, is
    #  calculated.
    def execute_validation(self):
        self._set_mode2exec()
        self._validLoss = 0

        if self._dataLoaderValid.__len__() == 0:
            self._logger.DisplayMessage('Empty validation set')
            self._validLoss = 0
        else:
            for batch_idx, (data, label) in enumerate(self._dataLoaderValid):
                self._input = Variable(data).float()
                self._label = Variable(label).float()
                if self._enabCuda:
                    self._input = self._input.cuda()
                    self._label = self._label.cuda()

                # evaluate based on specified routine
                self._forward()
                self._evaluate()
                self._validLoss += self._loss.data[0]

            self._validLoss /= (batch_idx + 1)

            message = 'Validation set loss: {:.8f}'.format(self._validLoss)
            if self._enabSave:
                self._logger.DisplayMessage(message)
            else:
                print(message)

    ## @brief Initialize the model for training
    #
    #  The functions needs to be called before the training process.
    #
    #  Simple example to perform a training:
    #
    #             model = model_of_choice()
    #             model.initialize_training()
    #             for epoch in range(cntEpochs):
    #                 model.execute_training()
    #                 model.execute_validation()
    def initialize_training(self):
        self._create_optimizers()
        self._create_data_loader()

    def _set_mode2train(self):
        pass

    def _set_mode2exec(self):
        pass

    ## @brief Offers serialization
    #
    #  The subclass may override this function to serialize its internal structures
    def save_model_state(self, path):
        pass

    ## @brief Offers deserialization
    #
    #  The subclass may override this function to deserialize itself
    def load_model_state(self, path):
        pass

    ## @brief Store meta data
    #
    #  Write meta data about the current model to the desired path
    def save_metadata(self, path):
        # create an ini file
        config = configparser.ConfigParser()
        config['General'] = {}

        # get the childs config
        modelConfig = self._get_metadata_impl()
        for key in modelConfig:
            config[key] = {}
            for entry in modelConfig[key]:
                config[key][entry] = str(modelConfig[key][entry])

        config['General']['ModelName'] = self.name()
        # write the file
        with open(os.path.join(path, 'ModelConfig.ini'), 'w') as configfile:
            config.write(configfile)

    ## @brief Load meta data
    #
    #  Try to load meta data from the specified path and configure itself accordingly.
    def load_metadata(self, path):
        if self._enab_debug_output:
            print("Loading model meta data:")
        config = configparser.ConfigParser()
        config.read(os.path.join(path, 'ModelConfig.ini'))

        # if 'modelname' in config['General']:
        #     print("Model name: {}".format(config['General']['modelname']))

        # let the child know about the config
        self._set_metadata_impl(config)

    # ___________________________________________________________________________________________________________________
    #
    # Implementation details

    ## @brief The forward pass
    #
    #  The subclass needs to perform the forward pass through its internal network structure and place the result in
    #  self._output. All internal networks are expected to be implemented using pytorch. Thus, self._output is expected
    #  to be a pytorch tensor.
    #
    #  For a simple example, the BDNN class may be considered.
    #
    #  @note    This function must be implemented by the subclass
    #  @warning Since the underlying network training process is dependent on the auto_grad functionality of pytorch,
    #           all possible computations need to performed within pytorch. If not, the remaining functionality will
    #           still be given but training the model will no longer be possible.
    def _forward(self):
        pass

    ## @brief The back propagation
    #
    #  The subclass needs to perform the back propagation through its internal network structure.
    #
    #  For a simple example, the BDNN class may be considered.
    #
    #  @note    This function must be implemented by the subclass
    def _backward(self):
        pass

    ## @brief The forward pass
    #
    # The functions computes the loss or error metric of choice between the computed output, self._output, and the
    # corresponding label, self._label.
    #
    #  @note    This function needs to be implemented by the subclass, if the funcionality of execute_validation is
    #           desired
    def _evaluate(self):
        pass

    ## @brief Perform the parameter optimization process in training mode
    def _optimize_params(self):
        self._forward()
        self._backward()

    ## @brief Creates the optimizers of choice
    #
    # The subclass may define in this function the optimizers it would like to use.
    #
    # Example:
    #           self._optimizer = optim.Adam(self._network.parameters(), lr=self._learnRate)
    def _create_optimizers(self):
        pass

    ## @brief Create the data loader of choice
    #
    #  The subclass may define in this function the data loading
    def _create_data_loader(self):
        pass

    ## @brief Offers meta data storage
    #
    #  The subclass may override this function to store meta data about itself, e.g. details on network architecture
    def _get_metadata_impl(self):
        pass

    ## @brief Offers meta data loading
    #
    #  The subclass may override this function to configure itself according to the stored metadata
    def _set_metadata_impl(self, config):
        pass

#----------------------------------------------------------------------------------------------------------------------
## @brief The base model for DNNs
#
#   _criterion, _optimizer
#  All subclasses need to specify their _dataLoaderTrain, _network,
#
#-----------------------------------------------------------------------------------------------------------------------
class BDNN(BaseModel):
    def __init__(self):
        super(BDNN, self).__init__()
        self._network = [] # to be defined by the base class and need to be derived from torch.nn.Module
        self._optimizer = []

        if 0:
            self._name_criterion = 'MSE'
            self._criterion = torch.nn.MSELoss() # default criterion
        else:
            self._name_criterion = 'MPAD'
            self._criterion = CustomLoss.LossMPAD()
        self._patchSize = 32
        self._batchSize = 5

    def _create_optimizers(self):
        self._optimizer = optim.Adam(self._network.parameters(), lr=self._learnRate)
        if self._enabCuda:
            self._network.cuda()

    def _set_mode2train(self):
        self._network.train()

    def _set_mode2exec(self):
        self._network.eval()

    def _backward(self):
        # clear grad
        self._optimizer.zero_grad()

        # compute the loss
        self._loss = self._criterion(self._output, self._label)
        self._loss.backward()
        self._trainLoss += self._loss[0].data

        # update parameters
        self._optimizer.step()

    def _evaluate(self):
        self._loss = self._criterion(self._output, self._label)

    def _forward(self):
        self._output = self._network(self._input)

    def get_loss(self):
        return OrderedDict([('loss', self._loss.data[0])])

    def save_model_state(self, path):
        # save the actual network
        SaveNet(path, self._network, "network", self._cntEpoch)

    def load_model_state(self, path, epoch):
        filename = os.path.join(path, "network_epoch{}.mod".format(epoch))
        self._network.load_state_dict(torch.load(filename))

        if self._enabCuda:
            self._network.cuda()
