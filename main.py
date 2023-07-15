import sys
import tkinter as tk
import os
import customtkinter

current_dir = os.path.dirname(os.path.abspath(__file__))
resources_path = os.path.join(current_dir, "resources")


def modo_normal():
    os.system("python " + os.path.join(resources_path, "modo_normal.py"))


def modo_juego():
    os.system("python " + os.path.join(resources_path, "modo_juego.py"))


# Crear la ventana principal
window = customtkinter.CTk()
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

logo_path = os.path.join(resources_path, "logo.ico")
window.iconbitmap(logo_path)
window.wm_iconbitmap(logo_path)

modo_Normal = customtkinter.CTkButton(window, text="Modo libre", command=modo_normal)
modo_Normal.pack(pady=10)

modo_Juego = customtkinter.CTkButton(window, text="Modo de juego", command=modo_juego)

modo_Juego.pack(pady=10)

window.mainloop()
