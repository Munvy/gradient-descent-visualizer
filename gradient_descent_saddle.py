import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- 1. PARAMETRY OPTYMALIZACJI ---
x = 2.0             # Punkt startowy na osi X
y = 0.05            # Punkt startowy na osi Y (blisko grzbietu, by pokazać ucieczkę)
alfa = 0.1          # Learning Rate (współczynnik uczenia)
iters = 15          # Liczba kroków (ograniczona, by uniknąć eksplozji gradientu)

historia_x = []
historia_y = []

# --- 2. PĘTLA ALGORYTMU GRADIENT DESCENT ---
for i in range(iters):
    # Obliczamy pochodne cząstkowe (gradient) dla funkcji f(x,y) = x^2 - y^2
    grad_x = 2 * x
    grad_y = -2 * y

    # Aktualizacja parametrów (ruch w kierunku przeciwnym do gradientu)
    x = x - alfa * grad_x
    y = y - alfa * grad_y

    # Zapisujemy historię do późniejszej wizualizacji
    historia_x.append(x)
    historia_y.append(y)

print(f"Finalne współrzędne po {iters} krokach: x = {x:.4f}, y = {y:.4f}")

# --- 3. PRZYGOTOWANIE TERENU DO WIZUALIZACJI ---
x_visual = np.linspace(-3, 3, 100)
y_visual = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_visual, y_visual)
Z = X**2 - Y**2

# --- 4. TWORZENIE WYKRESU 3D ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Rysowanie powierzchni siodła
surface = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.6)

# Obliczanie wysokości Z dla punktów, w których była kropka (historia)
historia_z = [px**2 - py**2 for px, py in zip(historia_x, historia_y)]

# Nakładanie ścieżki algorytmu na powierzchnię
ax.plot(historia_x, historia_y, historia_z, color='black', linewidth=3, marker='o', label='Ścieżka Gradient Descent')

# --- 5. DOPRACOWANIE WYGLĄDU ---
ax.set_title("Wizualizacja ucieczki z punktu siodłowego (Saddle Point)")
ax.set_xlabel("Parametr X")
ax.set_ylabel("Parametr Y")
ax.set_zlabel("Koszt (Błąd)")

# Dodanie rzutu poziomicowego na dno (dla lepszego efektu głębi)
ax.contour(X, Y, Z, zdir='z', offset=-10, cmap='coolwarm')
ax.set_zlim(-10, 10) # Stały zakres osi Z

plt.legend()
plt.tight_layout()
plt.show()
