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
window.title("CamSingGT")
window.geometry("400x200")

window.update()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window_width = window.winfo_width()
window_height = window.winfo_height()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"+{x}+{y}")

# Estilo de los botones
button_style = {
    "font": ("Helvetica", 14),
    "bg": "#3E82C7",
    "fg": "white",
    "relief": "raised",
}

logo_path = os.path.join(resources_path, "logo.ico")
window.iconbitmap(logo_path)
window.wm_iconbitmap(logo_path)

modo_Normal = tk.Button(
    window, text="Modo libre", command=modo_normal, **button_style
)
modo_Normal.pack(pady=10)

modo_Juego = tk.Button(
    window, text="Modo de juego", command=modo_juego, **button_style
)
modo_Juego.pack(pady=10)

window.mainloop()
