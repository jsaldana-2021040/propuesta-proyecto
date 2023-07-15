import sys
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter

class modo_normal(customtkinter.CTk):
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
window.geometry("800x600")
window.after(0, lambda:window.state('zoomed'))

logo_path = os.path.join(resources_path, "logo.ico")
window.iconbitmap(logo_path)
window.wm_iconbitmap(logo_path)

label = tk.Label(window)
label.place(x=360, y=100)

predicted_label = customtkinter.CTkLabel(window, font=("Helvetica", 30))
predicted_label.pack()

cap = cv2.VideoCapture(0)

# Iniciar la actualización del video
update_frame()

# Iniciar el bucle de eventos de Tkinter
window.mainloop()

# Liberar la captura de video y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
