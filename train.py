#!/usr/bin/env python3

from networks.Models import BaseModel_ICVLChallenge
from networks import BasicNetworks
import os
import torch.optim as optim

# configuration
track = 'Clean'
depthUnet = 5
cntEpochs = 5           # epochs to train
storagePath = '/home/temp/stiebel/rgb2spec/'
storagePath = storagePath + track + '/training/'.format(depthUnet)

model = BaseModel_ICVLChallenge(track)
model._network = BasicNetworks.GenUNetNoPooling(cnt_downs=depthUnet)

if not os.path.exists(storagePath):
    os.makedirs(storagePath)
model.enab_logging(path=storagePath)
model.save_metadata(storagePath)

# Training
model.initialize_training()
for epoch in range(1, cntEpochs + 1):
    model.execute_training()
    model.execute_validation()
    model.save_model_state(storagePath)

model._learnRate = 0.0000003
model._optimizer = optim.SGD(model._network.parameters(), lr=model._learnRate, momemtum=0.9, nesterov=1)
for epoch in range(1, cntEpochs + 1):
    model.execute_training()
    model.execute_validation()
    model.save_model_state(storagePath)



