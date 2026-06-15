document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("paintCanvas");
    const ctx = canvas.getContext("2d");
    const clearBtn = document.getElementById("clearBtn");
    const predictBtn = document.getElementById("predictBtn");
    const mainResult = document.getElementById("mainResult");
    const confidenceDiv = document.getElementById("confidence");
    const chartBars = document.getElementById("chartBars");

    let drawing = false;

    // Configuração do traço do pincel de desenho
    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#ffffff"; 

    // Renderiza a estrutura inicial dos gráficos de probabilidade
    for (let i = 0; i < 10; i++) {
        chartBars.innerHTML += `
            <div class="bar-container">
                <span class="bar-label">${i}</span>
                <div class="bar-outer"><div class="bar-inner" id="bar-${i}"></div></div>
                <span id="val-${i}">0%</span>
            </div>
        `;
    }

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    canvas.addEventListener("mousedown", (e) => {
        drawing = true;
        const pos = getMousePos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!drawing) return;
        const pos = getMousePos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    });

    window.addEventListener("mouseup", () => { drawing = false; });

    function limparCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        mainResult.innerText = "-";
        confidenceDiv.innerText = "Confiança: --%";
        for (let i = 0; i < 10; i++) {
            document.getElementById(`bar-${i}`).style.width = "0%";
            document.getElementById(`val-${i}`).innerText = "0%";
        }
    }

    async function classificarDigito() {
        const dataURL = canvas.toDataURL("image/png");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL })
            });

            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            mainResult.innerText = data.digit;
            confidenceDiv.innerText = `Confiança: ${(data.confidence * 100).toFixed(2)}%`;

            data.probabilities.forEach((prob, i) => {
                const percentual = (prob * 100).toFixed(0);
                document.getElementById(`bar-${i}`).style.width = `${percentual}%`;
                document.getElementById(`val-${i}`).innerText = `${percentual}%`;
            });

        } catch (err) {
            console.error("Erro na comunicação com o servidor:", err);
        }
    }

    clearBtn.addEventListener("click", limparCanvas);
    predictBtn.addEventListener("click", classificarDigito);

    // Atalhos de teclado requisitados
    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") limparCanvas();
        if (e.key === " ") {
            e.preventDefault(); 
            classificarDigito();
        }
    });
});