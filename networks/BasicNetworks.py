import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------------------------------
## @brief A UNet as discrimator
#
#  More details.
# ----------------------------------------------------------------------------------------------------------------------
class GenUNetNoPooling(torch.nn.Module):
    def __init__(self, cnt_downs = 10):
        super(GenUNetNoPooling, self).__init__()
        self._name = 'GenUNetNoPooling'

        depth = 0

        # construct the unet structure
        self._use_preproc = 0
        self._cntFirstFilter = 32
        self._cnt_downs = cnt_downs
        self._cnt_output_chan = 31
        self._cnt_input_chan = 3
        self._max_filter_mul = 4

        self._cntFilterPP = 0   # amount of post processing
        self._useBias = 1
        self._useSelu = 0
        self._useDropout = 0
        self._dropoutRate = 0.1

        self.CreateModel()

    def CreateModel(self):
        cff = self._cntFirstFilter

        if self._use_preproc:
            self.preproc = nn.Conv2d(self._cnt_input_chan, self._cnt_input_chan, kernel_size=5, stride=1,
                                     padding=2)
        if self._max_filter_mul == 4:
            submodule = UNetDownSampleBlockNoPooling(cff * 4, cff * 4, cntChannInput=None, submodule=None,
                                                     innermost=True, useSelu=self._useSelu, cntPP=self._cntFilterPP,
                                                     useDropout=self._useDropout, useBias=self._useBias)

            for i in range(self._cnt_downs - 4):
                submodule = UNetDownSampleBlockNoPooling(cff * 4, cff * 4, cntChannInput=None, cntPP=self._cntFilterPP,
                                                         submodule=submodule,useSelu=self._useSelu,
                                                         useDropout=self._useDropout, useBias=self._useBias)

            submodule = UNetDownSampleBlockNoPooling(cff * 2, cff * 4, cntChannInput=None, cntPP=self._cntFilterPP,
                                                     submodule=submodule, useSelu=self._useSelu,
                                                     useDropout=self._useDropout, useBias=self._useBias)
            submodule = UNetDownSampleBlockNoPooling(cff, cff * 2, cntChannInput=None, cntPP=self._cntFilterPP,
                                                     submodule=submodule, useSelu=self._useSelu,
                                                     useDropout=self._useDropout, useBias=self._useBias)
            submodule = UNetDownSampleBlockNoPooling(self._cnt_output_chan, cff, cntChannInput=self._cnt_input_chan,
                                                     submodule=submodule, outermost=True, cntPP=self._cntFilterPP,
                                                     useSelu=self._useSelu, useDropout=self._useDropout, useBias=self._useBias)
        elif self._max_filter_mul == 2:
            submodule = UNetDownSampleBlockNoPooling(cff * 2, cff * 2, cntChannInput=None, submodule=None,
                                                     innermost=True, useSelu=self._useSelu, cntPP=self._cntFilterPP,
                                                     useDropout=self._useDropout, useBias=self._useBias)

            for i in range(self._cnt_downs - 3):
                submodule = UNetDownSampleBlockNoPooling(cff * 2, cff * 2, cntChannInput=None, cntPP=self._cntFilterPP,
                                                         submodule=submodule, useSelu=self._useSelu,
                                                         useDropout=self._useDropout, useBias=self._useBias)
            submodule = UNetDownSampleBlockNoPooling(cff, cff * 2, cntChannInput=None, cntPP=self._cntFilterPP,
                                                     submodule=submodule, useSelu=self._useSelu,
                                                     useDropout=self._useDropout, useBias=self._useBias)
            submodule = UNetDownSampleBlockNoPooling(self._cnt_output_chan, cff, cntChannInput=self._cnt_input_chan,
                                                     submodule=submodule, outermost=True, cntPP=self._cntFilterPP,
                                                     useSelu=self._useSelu, useDropout=self._useDropout,
                                                     useBias=self._useBias)
        else:
            print("error creating model, invalid _max_filter_mul setting")

        self.model = submodule

    def forward(self, x):
        if self._use_preproc:
            xin = self.preproc(x)
            return self.model(xin)
        else:
            return self.model(x)

    def set_config(self, config):
        self._cntFirstFilter = int(config['cntFirstFilter'])
        self._cnt_input_chan = int(config['cntInputChannels'])
        self._cnt_output_chan = int(config['cntOutputChannels'])
        self._cnt_downs = int(config['cntDownsampling'])
        self._useSelu = int(config['useSelu'])

        self._useDropout = int(config['useDropout'])
        self._dropoutRate = float(config['dropoutRate'])

        if 'cntPostproc' in config:
            self._cntFilterPP = int(config['cntPostproc'])
        else:
            self._cntFilterPP = 0

        if 'maxFilterMul' in config:
            self._max_filter_mul = int(config['maxFilterMul'])
        else:
            self._max_filter_mul = 4

        if 'useBias' in config:
            self._useBias = int(config['useBias'])
        else:
            self._useBias = 1

        if 'usePreproc' in config:
            self._use_preproc = int(config['usePreproc'])
        else:
            self._use_preproc = 0

        for key in config:
            print(key + ": " + config[key])

        # create the model accordingly
        self.CreateModel()

    def get_config(self):
        config = {}
        config['name'] = self._name
        config['cntFirstFilter'] = self._cntFirstFilter
        config['cntInputChannels'] = self._cnt_input_chan
        config['cntOutputChannels'] = self._cnt_output_chan
        config['cntDownsampling'] = self._cnt_downs
        config['useSelu'] = self._useSelu
        config['maxFilterMul'] = self._max_filter_mul
        config['useDropout'] = self._useDropout
        config['useBias'] = self._useBias
        config['dropoutRate'] = self._dropoutRate
        config['usePreproc'] = self._use_preproc
        config['cntPostproc'] = self._cntFilterPP
        return config

# -----------------------------------------------------------------------------------------------------------------------
## @brief
#
#  More details.
# -----------------------------------------------------------------------------------------------------------------------
class UNetDownSampleBlockNoPooling(nn.Module):
    def __init__(self, outer_nc, inner_nc, cntChannInput=None, submodule=None, outermost=False, innermost=False,
                 cntPP=0, useSelu=0, useDropout=0, useBias=True):
        super(UNetDownSampleBlockNoPooling, self).__init__()
        self.outermost = outermost

        dropout = 0.1
        kernSize = 3

        if cntChannInput is None:
            cntChannInput = outer_nc

        # ... define the downconvolution
        downconv = nn.Conv2d(cntChannInput, inner_nc, kernel_size=kernSize, stride=1, padding=0, bias=useBias)
        if useSelu:
            downrelu = nn.SELU()
            uprelu = nn.SELU()
        else:
            downrelu = nn.LeakyReLU(0.2, True)
            uprelu = nn.ReLU(True)

        down = [downconv, downrelu]


        # ... define postprocessing filters
        if cntPP:
            pf1 = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=kernSize, stride=1, padding=1, bias=useBias)
            if useSelu:
                postproc = [pf1, nn.SELU()]
                if useDropout:
                    postproc.append(nn.AlphaDropout(dropout))
            else:
                postproc = [pf1, nn.ReLU(True)]

            for ind in range(1, cntPP):
                pf = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=useBias)
                postproc.append(pf)
                if useSelu:
                    postproc.append(nn.SELU())
                    if useDropout:
                        postproc.append(nn.AlphaDropout(dropout))
                else:
                    postproc.append(nn.ReLU(True))
        else:
            postproc = []

        # ____________________________________________________________________________________
        # 2.) create the actual module parts

        # ... handle the outermost case
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernSize, stride=1, padding=0,
                                        output_padding=0, bias=useBias)
            if useSelu:
                up = [upconv, nn.SELU()]
                if useDropout:
                    up.append(nn.AlphaDropout(dropout))
            else:
                up = [upconv, nn.ReLU()]
            inmodel = [submodule] + up

            # self.finalProc = nn.Conv2d(inner_nc+cntChannInput, outer_nc, kernel_size=kernSize, stride=1, padding=1)

        # ... handle the innermost case
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=kernSize, stride=1, padding=0, output_padding=0,
                                        bias=useBias)

            if useSelu:
                up = [upconv, nn.SELU()]
                if useDropout:
                    up.append(nn.AlphaDropout(dropout))
            else:
                up = [upconv, nn.ReLU()]

            inmodel = up

        # ... handle the default case
        else:

            # depending on whether postprocessing filters are enabled, the upconvulotion gets a different amount of
            # input channels
            if cntPP:
                upconv = nn.ConvTranspose2d(outer_nc, outer_nc, kernel_size=kernSize, stride=1, padding=0,
                                            output_padding=0, bias=useBias)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernSize, stride=1, padding=0,
                                        output_padding=0, bias=useBias)
            if useSelu:
                up = [upconv, nn.SELU()]
                if useDropout:
                    up.append(nn.AlphaDropout(dropout))
            else:
                up = [upconv, nn.ReLU()]

            inmodel = [submodule] + postproc + up

        self.down = nn.Sequential(*down)
        self.inmodel = nn.Sequential(*inmodel)

    def forward(self, x):
        xdown = self.down(x)
        xproc = self.inmodel(xdown)

        if self.outermost:
            return xproc
        else:
            combined = torch.cat([x, xproc], 1)
            return combined