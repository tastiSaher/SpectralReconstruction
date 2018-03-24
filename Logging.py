import numpy as np
import os
import time

class Log():
    def __init__(self, storagepath):

        self.log_name = os.path.join(storagepath, 'log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def DisplayErrors(self, indBatch, totalSize, errors):
        message = '({}/{}) '.format (indBatch, totalSize)
        for k, v in errors.items():
            message += ', %s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def DisplayEpoch(self, epoch):
        message = "Epoch {}".format(epoch)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def DisplayMessage (self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    # def PlotLog(self):