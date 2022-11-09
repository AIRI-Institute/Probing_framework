import torch

from probing.utils import *


class LogReg(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
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
        self, input_dim: int, num_classes: int, hidden_size: int, dropout_rate: float
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


class LinearVariational(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        parent, 
        bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.include_bias = bias        
        self.parent = parent
        
        if getattr(parent, "accumulated_kl_div", None) is None:
            if getattr(parent.parent, "accumulated_kl_div", None) is None:
                parent.accumulated_kl_div = 0
            else: 
                parent.accumulated_kl_div = parent.parent.accumulated_kl_div
            
        self.w_mu = torch.nn.Parameter(
            torch.FloatTensor(in_features, out_features)
            .normal_(mean=0, std=0.001)
            .to(self.device)
        )
        self.w_p = torch.nn.Parameter(
            torch.FloatTensor(in_features, out_features)
            .normal_(mean=0, std=0.001)
            .to(self.device)
        )

        if self.include_bias:
            self.b_mu = torch.nn.Parameter(torch.zeros(out_features))
            self.b_p = torch.nn.Parameter(torch.zeros(out_features))

    @staticmethod
    def _reparameterize(self, mu, p):
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)

    def forward(self, x):
        w = self._reparameterize(self.w_mu, self.w_p)
        
        if self.include_bias: 
            b = self._reparameterize(self.b_mu, self.b_p)
        else: 
            b = 0
            
        z = torch.matmul(x, w) + b
        
        self.parent.accumulated_kl_div += kl_divergence(w, self.w_mu, self.w_p).item()
        if self.include_bias: 
            self.parent.accumulated_kl_div += kl_divergence(
                b, self.b_mu, self.b_p).item()
        return z


class MDLLinearModel(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.kl_loss = KL
        self.layers = torch.nn.Sequential(
            LinearVariational(input_dim, num_classes, self.kl_loss, device)
        )

    @property
    def accumulated_kl_div(self):
        # assert self.variational
        return self.kl_loss.accumulated_kl_div
    
    def reset_kl_div(self):
        # assert self.variational
        self.kl_loss.accumulated_kl_div = 0
            
    def forward(self, x):
        # for l in self.layers.modules(): print(list(l.parameters()))
        return self.layers(x)
