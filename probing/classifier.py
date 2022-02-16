import torch


class LogReg(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return x


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_hidden: int=256,
        dropout_rate: float=0.2
    ):
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.fc1 = torch.nn.Linear(self.input_dim, self.num_hidden)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.activation = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(self.num_hidden, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
