import numpy as np

def calculate(list_input):
    # Validar longitud
    if len(list_input) != 9:
        raise ValueError("La lista debe contener nueve números")

    # Convertir a matriz 3x3
    arr = np.array(list_input).reshape(3, 3)

    # Calcular métricas
    calculations = {
        'mean': [
            arr.mean(axis=0).tolist(),
            arr.mean(axis=1).tolist(),
            arr.mean().item()
        ],
        'variance': [
            arr.var(axis=0).tolist(),
            arr.var(axis=1).tolist(),
            arr.var().item()
        ],
        'standard deviation': [
            arr.std(axis=0).tolist(),
            arr.std(axis=1).tolist(),
            arr.std().item()
        ],
        'max': [
            arr.max(axis=0).tolist(),
            arr.max(axis=1).tolist(),
            arr.max().item()
        ],
        'min': [
            arr.min(axis=0).tolist(),
            arr.min(axis=1).tolist(),
            arr.min().item()
        ],
        'sum': [
            arr.sum(axis=0).tolist(),
            arr.sum(axis=1).tolist(),
            arr.sum().item()
        ]
    }

    return calculations
