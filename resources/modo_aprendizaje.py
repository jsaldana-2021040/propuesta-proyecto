import sys
import string
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from random import choice
import customtkinter

import warnings

warnings.filterwarnings('ignore')


class modo_juego(customtkinter.CTk):
    def __init__(self):
        super().__init__()


resources_path = getattr(sys, "_MEIPASS", "./resources")

model_dict = pickle.load(open(os.path.join(resources_path, "model.p"), "rb"))
model = model_dict["model"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[-1] = "Unknown"

letters = [chr(65 + i) for i in range(26)]

current_letter = choice(letters)
saved_letters = []  # Lista para almacenar las letras guardadas


def predict_character(frame):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux[:42])])

        predicted_character = labels_dict.get(int(prediction[0]), labels_dict[-1])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame,
            predicted_character,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        return frame, predicted_character
    else:
        return frame, ""


def update_frame():
    ret, frame = cap.read()

    if frame is not None:
        processed_frame, predicted_character = predict_character(frame)

        image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        image = ImageTk.PhotoImage(image)

        label.configure(image=image)
        label.image = image

        predicted_label.configure(text=predicted_character)

    label.after(1, update_frame)


window = customtkinter.CTk()
window.title("CamSingGT")
window.geometry("880x655")
window.update()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window_width = window.winfo_width()
window_height = window.winfo_height()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"+{x}+{y}")

# Creación de menú para la opción de cerrar ventana
menu_frame = customtkinter.CTkFrame(window)
menu_frame.pack(side="top", fill="x")

opciones = tk.Menubutton(
    menu_frame,
    text="Opciones",
    background="#2b2b2b",
    foreground="white",
    activeforeground="black",
    activebackground="gray52",
)

menu_superior = tk.Menu(opciones, tearoff=0)
menu_superior.add_command(
    label="Salir",
    command=window.quit,
    background="#2b2b2b",
    foreground="white",
    activeforeground="black",
    activebackground="gray52",
)


def get_letter_by_number(number):
    alphabet_string = string.ascii_uppercase
    return alphabet_string[number - 1]


opciones.config(menu=menu_superior)
opciones.pack(side="left")

logo_path = os.path.join(resources_path, "logo.ico")
window.iconbitmap(logo_path)
window.wm_iconbitmap(logo_path)

frame = customtkinter.CTkFrame(master=window, corner_radius=8)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Place elements using grid layout
label = tk.Label(frame)
label.grid(row=0, column=0, columnspan=3, pady=20, padx=20)

predicted_label = customtkinter.CTkLabel(frame, font=("Helvetica", 40))
predicted_label.grid(row=1, column=0, columnspan=3, pady=5)

abc_img = ImageTk.PhotoImage(
    Image.open(resources_path + "/letras/A.png"), size=(400, 400)
)

number = 1

def next_letter():
    global number, current_letter
    number += 1
    current_letter = get_letter_by_number(number)
    if number == 26:
        next_button.grid_forget()
    if number > 1:
        previous_button.grid(row=2, column=0, padx=0, pady=5)  # Show the "previous" button
    new_image = ImageTk.PhotoImage(
        Image.open(resources_path + f"/letras/{current_letter}.png"), size=(400, 400)
    )
    image_label.configure(image=new_image)
    image_label.image = new_image

def previous_letter():
    global number, current_letter
    number -= 1
    current_letter = get_letter_by_number(number)
    if number == 1:
        previous_button.grid_forget()
    if number < 26:
        next_button.grid(row=2, column=4, padx=60, pady=5)  # Show the "next" button
    new_image = ImageTk.PhotoImage(
        Image.open(resources_path + f"/letras/{current_letter}.png"), size=(400, 400)
    )
    image_label.configure(image=new_image)
    image_label.image = new_image

frame.columnconfigure(3, weight=1)
image_label = customtkinter.CTkLabel(
    frame,
    image=abc_img,
    text="Haz la siguiente letra:",
    font=("Helvetica", 30),
    compound="bottom",
    pady=30,
)
image_label.grid(row=0, column=3, columnspan=3, pady=0, sticky="nsew")

previous_button = customtkinter.CTkButton(
    frame,
    text="Anterior",
    anchor="center",
    width=219,
    font=("Roboto", 20),
    height=40,
    command=previous_letter,
)
previous_button.grid(row=2, column=0, padx=0, pady=5) 
previous_button.grid_forget()

next_button = customtkinter.CTkButton(
    frame,
    text="Siguiente",
    anchor="center",
    width=219,
    font=("Roboto", 20),
    height=40,
    command=next_letter,
)
next_button.grid(row=2, column=4, padx=60, pady=5)

window.state('zoomed')

cap = cv2.VideoCapture(0)

# Iniciar la actualización del video
update_frame()

# Iniciar el bucle de eventos de Tkinter
window.mainloop()

# Liberar la captura de video y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
