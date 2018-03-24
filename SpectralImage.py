import numpy as np, h5py

class MSpecImage():
    def __init__(self):
        self.data = []
        self.rows = 0
        self.cols = 0

        self.patchstride = 2

    def LoadICVLSpectral(self, filename):
        f = h5py.File(filename)
        self.data = np.array(f['rad'])
        self.data = np.swapaxes(self.data, 0, 2)
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]

    def GetCntPossiblePatches(self, height_k, width_k):
        cntPatchesW = (self.cols-width_k) / int(width_k / 2)
        cntPatchesH = (self.rows - height_k) / int(height_k / 2)
        return int(cntPatchesW) * int(cntPatchesH)

    def GetCntPossiblePatchesAll(self, height_k, width_k):
        cntPatchesW = (self.cols - width_k)
        cntPatchesH = (self.rows - height_k)
        return int(cntPatchesW) * int(cntPatchesH)

    def GetPatchStart(self, indPatch, height_k, width_k):
        cntPatchesW = int((self.cols - width_k) / int(width_k / 2))

        deltaR = int(height_k / 2)
        deltaC = int(width_k / 2)

        indR = int(indPatch / cntPatchesW)
        indC = indPatch - (indR * cntPatchesW)

        return indR * deltaR, indC * deltaC

    def GetPatchStartAll(self, indPatch, height_k, width_k):
        cntPatchesW = (self.cols - width_k)

        indR = int(indPatch / cntPatchesW)
        indC = indPatch - (indR * cntPatchesW)

        return indR, indC

