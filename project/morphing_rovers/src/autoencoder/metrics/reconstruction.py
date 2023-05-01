import numpy as np


class ReconstructionMetric:
    def __init__(self, mode):
        self.mode = mode
        self.value = 0.0
        self.n_batch = 0

    def update(self, pred, target):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        self.value += np.mean((pred-target)**2)

        self.n_batch += 1

    def evaluate(self, global_step):
        if self.n_batch > 0:
            score = self.value / self.n_batch
        else:
            score = -float('inf')

        return score

