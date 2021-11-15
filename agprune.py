import torch
from torch.nn.utils.prune import BasePruningMethod
from torch import autograd

class agprune(BasePruningMethod):
    def __init__(self, initial_sparsity, final_sparsity, current_epoch, starting_epoch, ending_epoch, freq):
        super(agprune, self).__init__()
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.current_epoch = current_epoch
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        self.freq = freq

    def AgpPruningRate(self, initial_sparsity, final_sparsity, current_epoch, starting_epoch, ending_epoch, freq):
        """A pruning-rate scheduler per https://arxiv.org/pdf/1710.01878.pdf.
        """
        span = ((ending_epoch - starting_epoch - 1) // freq) * freq
        #print(span,'111111111')
        # span = ending_epoch - starting_epoch
        assert span > 0
        target_sparsity = (final_sparsity +
                            (initial_sparsity-final_sparsity) *
                            (1.0 - ((current_epoch-starting_epoch)/span))**3)

        return target_sparsity
    def compute_mask(self, t, default_mask):
        target_sparsity = self.AgpPruningRate(self.initial_sparsity, self.final_sparsity, self.current_epoch,
                                              self.starting_epoch, self.ending_epoch, self.freq)
        #with torch.no_grad():
        # partial sort
        tensor = t.data
        bottomk, _ = torch.topk(tensor.abs().view(-1),
                                int(target_sparsity * tensor.numel()),
                                largest=False,
                                sorted=True)
        threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
        mask = torch.gt(torch.abs(tensor), threshold).type(tensor.type())
        return mask


class AgPrune():
    def __init__(self, initial_sparsity, final_sparsity, current_epoch, starting_epoch, ending_epoch, freq):
        super(AgPrune, self).__init__()
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.current_epoch = current_epoch
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        self.freq =freq


    def AgpPruningRate(self):
        """A pruning-rate scheduler per https://arxiv.org/pdf/1710.01878.pdf.
        """
        span = ((self.ending_epoch - self.starting_epoch - 1) // self.freq) * self.freq
        assert span > 0
        target_sparsity = (self.final_sparsity +
                            (self.initial_sparsity-self.final_sparsity) *
                            (1.0 - ((self.current_epoch-self.starting_epoch)/span))**3)

        return target_sparsity

    def create_mask_threshold_criterion(self, tensor, threshold):
        with torch.no_grad():
            mask = torch.gt(torch.abs(tensor), threshold).type(tensor.type())
            return mask

    def set_param_mask_by_sparsity_target(self, tensor):
        target_sparsity = self.AgpPruningRate()
        with torch.no_grad():
            # partial sort
            tensor = tensor.data.clone()
            bottomk, _ = torch.topk(tensor.abs().view(-1),
                                    int(target_sparsity * tensor.numel()),
                                    largest=False,
                                    sorted=True)
            threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
            mask = self.create_mask_threshold_criterion(tensor, threshold)
            return mask

    def apply_prune(self, tensor):
        #tensor_data = tensor.data.clone()
        mask = self.set_param_mask_by_sparsity_target(tensor)
        return tensor * mask

class Prune(autograd.Function):
    """ Chooses the top edges for the forwards pass but allows gradient flow to all edges in the backwards pass"""

    @staticmethod
    def forward(ctx, weight, prune_rate):
        w = weight.data.clone()
        bottomk, _ = torch.topk(w.abs().view(-1),
                                int(prune_rate * w.numel()),
                                largest=False,
                                sorted=True)
        threshold = bottomk.data[-1]
        mask = torch.gt(torch.abs(w), threshold).type(w.type())
        ctx.save_for_backward(mask)
        return (weight.data * mask).type(w.type())

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[mask == 0] = 0
        return grad_input, None, None