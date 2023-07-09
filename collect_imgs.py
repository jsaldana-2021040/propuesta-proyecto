import cv2
import os

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 29
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    else:
        # Si la carpeta ya existe, se omitirá la clase y se mostrará un mensaje
        print("La carpeta {} ya existe. Se omitirá la clase {}".format(class_dir, j))
        continue

    print("Recolectando datos para la clase {}".format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame,
            "tulio hueco",
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord("q"):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, "{}.jpg".format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()