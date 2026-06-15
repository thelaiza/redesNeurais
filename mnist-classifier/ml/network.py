import torch
import torch.nn as nn

class DigitClassifierMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Flatten(),
            
            # Camada Oculta 1
            nn.Linear(784, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            
            # Camada Oculta 2
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Camada de Saída (10 classes: dígitos de 0 a 9)
            nn.Linear(200, 10)
        )

    def forward(self, x):
        return self.pipe(x)