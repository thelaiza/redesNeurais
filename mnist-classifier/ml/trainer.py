import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network import DigitClassifierMLP

def executar_treinamento():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando treinamento em: {device}")

    # Data Augmentation e Normalização padrão do MNIST
    transformacoes = transforms.Compose([
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Carga de dados
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transformacoes)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    model = DigitClassifierMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    model.train()
    for epoch in range(1, 16): # 15 Épocas é ideal para convergência rápida
        total_loss = 0
        corretos = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            corretos += pred.eq(target.view_as(pred)).sum().item()
            
        acc = 100. * corretos / len(train_loader.dataset)
        print(f"Época {epoch} -> Perda Média: {total_loss/len(train_loader):.4f} | Acurácia: {acc:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/melhor_modelo.pth")
    print("Treinamento finalizado! Pesos salvos em 'checkpoints/melhor_modelo.pth'")

if __name__ == "__main__":
    executar_treinamento()