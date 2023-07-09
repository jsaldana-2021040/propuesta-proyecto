import sys
import tkinter as tk
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
resources_path = os.path.join(current_dir, "resources")


def modo_normal():
    os.system("python " + os.path.join(resources_path, "modo_normal.py"))


def modo_juego():
    os.system("python " + os.path.join(resources_path, "modo_juego.py"))


# Crear la ventana principal
window = tk.Tk()
window.title("Hand Gesture Recognition")
window.geometry("400x200")

# Estilo de los botones
button_style = {
    "font": ("Helvetica", 14),
    "bg": "#3E82C7",
    "fg": "white",
    "relief": "raised",
}

# Crear un botón para ejecutar el código original
original_button = tk.Button(
    window, text="Modo libre", command=modo_normal, **button_style
)
original_button.pack(pady=10)

# Crear un botón para ejecutar el código modificado
modified_button = tk.Button(
    window, text="Modo de juego", command=modo_juego, **button_style
)
modified_button.pack(pady=10)

# Iniciar el bucle de eventos de Tkinter
window.mainloop()
