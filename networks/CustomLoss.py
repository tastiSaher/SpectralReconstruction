from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch


class LossMPAD(_Loss):
    def __init__(self, size_average=True):
        super(LossMPAD, self).__init__(size_average)

    def forward(self, input, target):
        # _assert_no_grad(target)
        d = 100 * torch.abs(1-torch.div(torch.squeeze(input), target))
        return torch.mean(d) if self.size_average else torch.sum(d)

class LossTest(_Loss):
    def __init__(self, size_average=True):
        super(LossMPAD, self).__init__(size_average)

    def forward(self, input, target):
        # _assert_no_grad(target)
        d = (input-target).mul(target)
        return torch.mean(d) if self.size_average else torch.sum(d)

# taken from pix2pix
class GANLoss(torch.nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = torch.nn.BCELoss()

    def get_target_tensor(self, input, isTargetReal):
        target_tensor = None
        if isTargetReal:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)