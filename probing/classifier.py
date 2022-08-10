import torch


class LogReg(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int
    ):
        super(LogReg, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(self.input_dim, self.num_classes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.activation(x)
        return x


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int,
        dropout_rate: float
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.activation = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
