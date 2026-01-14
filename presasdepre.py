import numpy as np
import matplotlib.pyplot as plt 

# Parámetros del sistema
ALPHA = 0.5
BETA = 0.02
GAMMA = 0.4
DELTA = 0.004

def sistema(t, estado):
    x, y = estado
    dxdt = ALPHA * x - BETA * x * y
    dydt = -GAMMA * y + DELTA * x * y
    return np.array([dxdt, dydt])

def rk4_step(f, t, h, y):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def simular(condiciones_iniciales, t_max=100, h=0.01):
    pasos = int(t_max / h)
    t = np.arange(0, t_max, h)

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)  # Plano fase
    ax2 = plt.subplot(1, 2, 2)  # Series temporales

    for ci in condiciones_iniciales:
        solucion = np.zeros((pasos, 2))
        solucion[0] = ci

        for i in range(pasos - 1):
            solucion[i+1] = rk4_step(sistema, t[i], h, solucion[i])

        # Plano de fase
        ax1.plot(solucion[:, 0], solucion[:, 1], label=f'CI: {ci}')

        # Solo graficar evolución temporal si está en cuadrante físico
        if ci[0] > 0 and ci[1] > 0:
            ax2.plot(t, solucion[:, 0], label=f'Presas {ci}')
            ax2.plot(t, solucion[:, 1], '--', label=f'Depredadores {ci}')

    # Decoración
    ax1.set_title('Plano de Fase')
    ax1.set_xlabel('Presas (x)')
    ax1.set_ylabel('Depredadores (y)')
    ax1.axhline(0, color='black')
    ax1.axvline(0, color='black')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Evolución Temporal')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Población')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Condiciones iniciales
condiciones = [
    [10, 30],
    [-10, 30],
    [-50, -10],
    [50, -10]
]

if name == "main":
    simular(condiciones)