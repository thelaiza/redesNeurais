import os
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request
from PIL import Image

# Importação relativa ajustando o path para encontrar o pacote ml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.network import DigitClassifierMLP

app = Flask(__name__, template_folder='../templates', static_folder='../static')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitClassifierMLP().to(device)
model_carregado = False

# Tenta carregar o arquivo de pesos gerado pelo trainer
path_pesos = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoints/melhor_modelo.pth'))
if os.path.exists(path_pesos):
    model.load_state_dict(torch.load(path_pesos, map_location=device))
    model.eval()
    model_carregado = True
    print("Pesos carregados com sucesso no Flask.")
else:
    print("Aviso: Treine o modelo executando 'python ml/trainer.py' primeiro.")

def transformar_imagem_base64(uri_base64):
    # Trata o cabeçalho do dado Base64 se houver
    if "," in uri_base64:
        uri_base64 = uri_base64.split(",")[1]
    
    dados_binarios = base64.b64decode(uri_base64)
    imagem = Image.open(io.BytesIO(dados_binarios)).convert("RGBA")
    
    # Cria uma máscara preta de fundo para colar o traço branco do canvas
    fundo_preto = Image.new("RGBA", imagem.size, (0, 0, 0, 255))
    fundo_preto.paste(imagem, mask=imagem.split()[3])
    tons_cinza = fundo_preto.convert("L")
    
    matriz = np.array(tons_cinza, dtype=np.float32)
    
    # Encontra as coordenadas limítrofes do caractere (Bounding Box)
    indices_desenhados = np.argwhere(matriz > 20)
    if indices_desenhados.size == 0:
        raise ValueError("Nenhum caractere detectado")
        
    ymin, xmin = indices_desenhados.min(axis=0)
    ymax, xmax = indices_desenhados.max(axis=0)
    
    # Recorta o dígito isolando o espaço em branco ao redor
    recorte = matriz[ymin:ymax+1, xmin:xmax+1]
    
    # Centraliza o recorte gerando um canvas quadrado proporcional
    maior_eixo = max(recorte.shape)
    padding = int(maior_eixo * 0.15)
    tamanho_final = maior_eixo + (padding * 2)
    
    canvas_quadrado = np.zeros((tamanho_final, tamanho_final), dtype=np.float32)
    start_y = (tamanho_final - recorte.shape[0]) // 2
    start_x = (tamanho_final - recorte.shape[1]) // 2
    canvas_quadrado[start_y:start_y+recorte.shape[0], start_x:start_x+recorte.shape[1]] = recorte
    
    # Redimensiona para o formato padrão 28x28
    img_28x28 = Image.fromarray(canvas_quadrado.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS)
    matriz_normalizada = np.array(img_28x28, dtype=np.float32) / 255.0
    
    # Normalização oficial do dataset MNIST
    matriz_normalizada = (matriz_normalizada - 0.1307) / 0.3081
    
    tensor_retorno = torch.tensor(matriz_normalizada).unsqueeze(0).unsqueeze(0).to(device)
    return tensor_retorno

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_carregado:
        return jsonify({"error": "Modelo não treinado ou indisponível"}), 503
        
    dados_requisicao = request.get_json()
    if not dados_requisicao or 'image' not in dados_requisicao:
        return jsonify({"error": "Dados ausentes"}), 400
        
    try:
        tensor_imagem = transformar_imagem_base64(dados_requisicao['image'])
    except ValueError:
        return jsonify({"error": "Desenhe algo no quadro antes de enviar"}), 400

    with torch.no_grad():
        logits = model(tensor_imagem)
        probabilidades = F.softmax(logits, dim=1)[0].cpu().numpy()
        
    predicao = int(probabilidades.argmax())
    confianca = float(probabilidades[predicao])
    
    return jsonify({
        "digit": predicao,
        "confidence": confianca,
        "probabilities": probabilidades.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)