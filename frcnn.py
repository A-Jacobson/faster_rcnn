from torch import nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dropout = TimeDistributed(nn.Dropout(0.5))
        self.fc1 = TimeDistributed(nn.Linear(512*7*7, 4096))
        self.fc2 = TimeDistributed(nn.Linear(4096, 4096))
        self.out_class = TimeDistributed(nn.Linear(4096, 21))
        self.out_regr = TimeDistributed(nn.Linear(4096, (4*20)))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), True)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), True)
        x = self.dropout(x)
        out_class = self.out_class(x)
        out_regr = self.out_regr(x)
        return out_class, out_regr



