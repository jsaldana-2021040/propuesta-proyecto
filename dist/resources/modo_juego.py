import sys
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from random import choice
from tkinter import messagebox
import customtkinter

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

letters = [chr(65 + i) for i in range(5)]

current_letter = choice(letters)
is_message_shown = False  # Variable para rastrear si el mensaje ya se ha mostrado
is_letter_saved = False  # Variable para rastrear si se ha guardado la letra
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
    global is_message_shown, is_letter_saved
    ret, frame = cap.read()

    if frame is not None:
        processed_frame, predicted_character = predict_character(frame)

        image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        image = ImageTk.PhotoImage(image)

        label.config(image=image)
        label.image = image

        predicted_label.configure(text=predicted_character)

        if (
            predicted_character.upper() == current_letter.upper()
            and not is_message_shown
            and predicted_character in letters
        ):
            if is_letter_saved:
                is_message_shown = True
                label.after(5000, show_success_message)
        elif (
            predicted_character.upper() != current_letter.upper()
            and predicted_character in letters
            and not is_message_shown
        ):
            if is_letter_saved:
                is_message_shown = True
    label.after(1, update_frame)


def show_failure_message():
    global is_message_shown, is_letter_saved
    if is_letter_saved:
        is_message_shown = True
        result = messagebox.askyesno(
            "¡Incorrecto!", "Has fallado. ¿Quieres intentarlo de nuevo?"
        )
        if result:
            save_letter()
            next_letter()
        else:
            window.quit()
            is_message_shown = False
            is_letter_saved = False


def show_success_message():
    global is_message_shown, is_letter_saved
    if is_letter_saved:
        is_message_shown = True
        success_window = tk.Toplevel(window)
        success_window.title("¡Correcto!")

        # Obtener el ancho y alto de la pantalla
        screen_width = success_window.winfo_screenwidth()
        screen_height = success_window.winfo_screenheight()

        # Calcular las coordenadas para centrar la ventana
        x = int((screen_width - 300) / 2)
        y = int((screen_height - 100) / 2)

        success_window.geometry(f"300x100+{x}+{y}")

        success_label = tk.Label(success_window, text="Has acertado la letra.")
        success_label.pack(pady=5)

        accept_button = customtkinter.CTkButton(
            success_window, text="Aceptar", 
            width=200,
            anchor="center",
            font=("Roboto", 20),
            height=40,
            command=success_window.destroy
        )
        accept_button.pack()

        save_letter()
        next_letter()
        is_message_shown = False
        is_letter_saved = False


def save_letter():
    global current_letter, is_letter_saved, saved_letters
    if current_letter not in saved_letters:
        saved_letters.append(current_letter)


def next_letter():
    global current_letter, is_message_shown, is_letter_saved
    is_message_shown = False
    is_letter_saved = False
    current_letter = choice(letters)
    instruction_label.configure(text="Haz la siguiente letra:")
    letter_label.configure(text=current_letter)


def save_gesture(event):
    global current_letter, is_letter_saved
    if event.keysym == "space":
        current_gesture = predicted_label.cget("text")
        if current_gesture.upper() in letters:
            is_letter_saved = True
            if is_letter_saved:
                result = messagebox.askyesno(
                    "Confirmación",
                    f"¿Está seguro de que hizo la letra '{current_gesture}'?",
                )
                if result:
                    if current_gesture.upper() == current_letter.upper():
                        save_letter()
                        show_success_message()
                    else:
                        show_failure_message()
        else:
            messagebox.showinfo("Advertencia", "No se ha detectado ningún gesto")
    is_letter_saved = False

window = customtkinter.CTk()
window.title("CamSingGT")
window.geometry("900x755")
window.update()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window_width = window.winfo_width()
window_height = window.winfo_height()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"+{x}+{y}")

#Creación de menú para la opción de cerrar ventana
menu_frame = customtkinter.CTkFrame(window)
menu_frame.pack(side='top', fill='x')

opciones = tk.Menubutton(menu_frame,
                         text="Opciones",
                         background="#2b2b2b",
                         foreground="white",
                         activeforeground="black",
                         activebackground="gray52")

menu_superior = tk.Menu(opciones, tearoff=0)
menu_superior.add_command(label="Salir", command=window.quit,
                         background="#2b2b2b",
                         foreground="white",
                         activeforeground="black",
                         activebackground="gray52")

opciones.config(menu=menu_superior)
opciones.pack(side="left")

logo_path = os.path.join(resources_path, "logo.ico")
window.iconbitmap(logo_path)
window.wm_iconbitmap(logo_path)

frame = customtkinter.CTkFrame(master=window, corner_radius=8)
frame.pack(pady=20, padx=20, fill="both", expand=True)

label = tk.Label(frame)
label.pack(pady=15)

predicted_label = customtkinter.CTkLabel(frame, font=("Helvetica", 40))
predicted_label.pack(pady=5)

instruction_label = customtkinter.CTkLabel(frame, font=("Helvetica", 20))
instruction_label.pack(pady=5)

letter_label = customtkinter.CTkLabel(frame, font=("Helvetica", 40), height=25)
letter_label.pack(pady=5)

backup_label = customtkinter.CTkLabel(frame, font=("Helvetica", 13), text="Preciona la tecla 'espacio' para evaluar.")
backup_label.pack(pady=3)

cap = cv2.VideoCapture(0)

# Vincular el evento <KeyRelease-Return> a la función save_gesture()
window.bind("<KeyRelease-space>", save_gesture)

# Iniciar la actualización del video
update_frame()

# Mostrar la primera letra
next_letter()

# Iniciar el bucle de eventos de Tkinter
window.mainloop()

# Liberar la captura de video y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
