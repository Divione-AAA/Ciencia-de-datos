import torch
import torch.nn as nn
import torch.optim as optim

# Datos: y = 2x
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Modelo: una capa lineal (y = wx + b)
modelo = nn.Linear(in_features=1, out_features=1)

# Función de pérdida (error cuadrático medio)
criterio = nn.MSELoss()

# Optimizador (algoritmo que mejora el modelo)
optimizador = optim.SGD(modelo.parameters(), lr=0.01)

# Entrenamiento
for epoca in range(1000):
    y_predicho = modelo(x)
    perdida = criterio(y_predicho, y)

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

    if epoca % 100 == 0:
        print(f"Época {epoca}, Pérdida: {perdida.item():.4f}")

# Probar con un valor nuevo
resultado = modelo(torch.tensor([[5.0]]))
print(f"Predicción para x=5: {resultado.item():.2f}")
