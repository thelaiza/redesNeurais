# 🧠 Classificação de Dígitos Manuscritos com Redes Neurais (MLP)

Este repositório contém uma solução completa para o problema de **Classificação de Dígitos Manuscritos** utilizando uma rede neural do tipo **Multilayer Perceptron (MLP)** desenvolvida em **PyTorch**. O projeto inclui um pipeline estruturado para o treino do modelo com o dataset **MNIST** e um ambiente interativo web construído em **Flask** para testar predições em tempo real usando o ponteiro do rato.

---

## Funcionalidades

* **Arquitetura Customizada:** Modelo MLP em PyTorch utilizando `BatchNorm1d`, `LeakyReLU` (0.1) e camadas de `Dropout` estrategicamente posicionadas para evitar o sobreajuste (*overfitting*).
* **Pipeline de Treino Estruturado:** Script de treino isolado com *Data Augmentation* dinâmico (`RandomRotation` e `RandomAffine`) para maior robustez na classificação.
* **Interface Web Interativa:** Canvas em HTML5 que permite desenhar dígitos livremente com o rato.
* **Feedback Visual Completo:** Gráfico de barras atualizado dinamicamente via JavaScript com as probabilidades calculadas pelo modelo para cada dígito (0 a 9).
* **Atalhos do Teclado:** Suporte a teclas de atalho para otimizar o fluxo de uso:
    * `Espaço`: Executa a classificação do desenho atual.
    * `Esc`: Limpa completamente o quadro de desenho.

---

## Estrutura do Projeto

A organização de pastas foi planeada para separar de forma limpa o ecossistema de Aprendizagem de Máquina (*Machine Learning*) dos ficheiros do servidor web:

    mnist-classifier/
    │
    ├── ml/
    │   ├── __init__.py         # Inicializador do pacote ml
    │   ├── network.py          # Arquitetura da Rede Neural (PyTorch)
    │   └── trainer.py          # Script de treino e validação com o MNIST
    │
    ├── server/
    │   ├── __init__.py         # Inicializador do pacote server
    │   └── main.py             # Servidor Web e API de inferência (Flask)
    │
    ├── templates/
    │   └── index.html          # Interface de utilizador em HTML5
    │
    ├── static/
    │   ├── css/
    │   │   └── style.css       # Estilização visual (Dark Mode moderno)
    │   └── js/
    │       └── canvas.js       # Lógica do Canvas, capturas e chamadas Fetch API
    │
    ├── checkpoints/
    │   └── melhor_modelo.pth   # Pesos do modelo guardados pós-treino
    │
    └── requirements.txt        # Dependências de bibliotecas do Python

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Machine Learning:** PyTorch, Torchvision
* **Processamento Numérico e Imagens:** NumPy, Pillow (PIL)
* **Back-end Web:** Flask
* **Front-end:** HTML5 Canvas, CSS3 (Modern Dark Palette), Vanilla JavaScript (ES6+)

---

## Como Executar o Projeto

Siga os passos abaixo para configurar o ambiente, treinar a rede neural e inicializar a interface web localmente.

### 1. Instalar Dependências

Primeiro, garanta que possui as bibliotecas necessárias instaladas:

    pip install -r requirements.txt

### 2. Treinar o Modelo

Antes de iniciar o servidor, é necessário gerar os pesos da rede neural através do script de treino. Este script irá descarregar automaticamente o dataset MNIST, aplicar as transformações geométricas e guardar os melhores pesos na pasta `checkpoints/`.

    python ml/trainer.py

*Nota: O script deteta automaticamente a presença de aceleração por GPU (CUDA) ou executará por omissão em CPU.*

### 3. Iniciar o Servidor Flask

Com o ficheiro `melhor_modelo.pth` gerado com sucesso, execute o servidor web:

    python server/main.py

A aplicação estará disponível no seu navegador através do endereço: **`http://localhost:5000`**

---

## Detalhes Técnicos da Rede Neural

O modelo implementado em `ml/network.py` segue uma abordagem diferente de arquiteturas tradicionais lineares básicas, melhorando a convergência através de normalização e funções de ativação mais avançadas:

1.  **Camada de Entrada:** *Flattening* transformando a matriz dimensional `28x28` para um vetor unidimensional de `784` neurónios.
2.  **Camada Oculta 1:** `Linear(784 → 400)` + `BatchNorm1d` + `LeakyReLU(0.1)` + `Dropout(p=0.25)`.
3.  **Camada Oculta 2:** `Linear(400 → 200)` + `BatchNorm1d` + `LeakyReLU(0.1)` + `Dropout(p=0.20)`.
4.  **Camada de Saída:** `Linear(200 → 10)` mapeando os scores brutos (*logits*) para as 10 classes possíveis.

No servidor Flask, os *logits* da imagem processada passam por uma função **Softmax**, convertendo as saídas em distribuições de probabilidade reais de 0% a 100% que alimentam a interface gráfica do utilizador.
