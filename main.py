import sys
import tkinter as tk
import os
import customtkinter
from PIL import Image, ImageTk

current_dir = os.path.dirname(os.path.abspath(__file__))
resources_path = os.path.join(current_dir, "resources")

def modo_normal():
    os.system("python " + os.path.join(resources_path, "modo_normal.py"))

def modo_juego():
    os.system("python " + os.path.join(resources_path, "modo_juego.py"))

def modo_aprendizaje():
    os.system("python " + os.path.join(resources_path, "modo_aprendizaje.py"))

#Cargando iconos para los buttons
icon_1 = customtkinter.CTkImage(Image.open(resources_path + "/free-mode-icon.png"), size=(50, 50))

icon_2 = customtkinter.CTkImage(Image.open(resources_path + "/game-mode-icon.png"), size=(50, 50))

icon_3 = customtkinter.CTkImage(Image.open(resources_path + "/learn-mode-icon.png"), size=(40, 40))

# Crear la ventana principal
window = customtkinter.CTk()
window.title("CamSingGT")
window.geometry("400x345")

window.update()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window_width = window.winfo_width()
window_height = window.winfo_height()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"+{x}+{y}")
window.resizable(False, False)

logo_path = os.path.join(resources_path, "logo.ico")
window.iconbitmap(logo_path)
window.wm_iconbitmap(logo_path)

frame = customtkinter.CTkFrame(master=window, corner_radius=8)
frame.pack(pady=20, padx=20, fill="both", expand=True)

modo_Normal = customtkinter.CTkButton(master=frame, 
                                      image=icon_1,
                                      text="Modo libre", 
                                      width=200,
                                      anchor="center",
                                      font=("Roboto", 20),
                                      height=40,
                                      command=modo_normal)
modo_Normal.pack(padx=20, pady=20)

modo_Juego = customtkinter.CTkButton(master=frame, 
                                     image=icon_2,
                                     text="Modo de juego", 
                                     anchor="center",
                                     width=219,
                                     font=("Roboto", 20),
                                     height=40,
                                     command=modo_juego)
modo_Juego.pack(padx=20, pady=20)

modo_Aprendizaje = customtkinter.CTkButton(master=frame, 
                                     image=icon_3,
                                     text="Modo de Aprendizaje", 
                                     anchor="center",
                                     width=219,
                                     font=("Roboto", 20),
                                     height=60,
                                     command=modo_aprendizaje)
modo_Aprendizaje.pack(padx=20, pady=20)

window.mainloop()
