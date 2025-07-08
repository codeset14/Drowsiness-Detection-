import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load Models ===
eye_model = load_model("face_eye_status_model.h5")
age_model = load_model("age_model.h5")
age_model.load_weights("age_model.weights.h5")

# === Labels ===
eye_labels = ['Closed', 'Open']
age_labels = ['0-12', '13-19', '20-35', '36-50', '51+']

# === Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Prediction Function ===
def predict_face_status(face_img):
    face_resized = cv2.resize(face_img, (64, 64))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_input = np.expand_dims(face_rgb / 255.0, axis=0)

    prob_eye = eye_model.predict(face_input)[0][0]
    eye_label = eye_labels[int(prob_eye > 0.5)]

    age_probs = age_model.predict(face_input)[0]
    age_idx = np.argmax(age_probs)
    age_label = age_labels[age_idx]
    age_conf = age_probs[age_idx]

    return eye_label, prob_eye, age_label, age_conf

# === Image Processing Handler ===
def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Unable to load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    awake_count = 0
    asleep_count = 0
    details_list = []

    for i, (x, y, w, h) in enumerate(faces):
        face_roi = img[y:y+h, x:x+w]

        try:
            eye_status, eye_conf, age_group, age_conf = predict_face_status(face_roi)
            is_awake = eye_status == 'Open'

            if is_awake:
                awake_count += 1
            else:
                asleep_count += 1

            details_list.append(f"Person {i+1}:\n  Eye: {eye_status}\n  Age: {age_group} ({age_conf:.2f})")

        except Exception as e:
            print("Prediction error:", e)

    # Resize image for display
    max_width, max_height = 800, 600
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (new_w, new_h))

    # Show image
    pil_img = Image.fromarray(resized_img)
    tk_img = ImageTk.PhotoImage(pil_img)
    panel.config(image=tk_img)
    panel.image = tk_img

    # Show popup summary
    if details_list:
        summary_msg = f"Total: {len(faces)}\nAwake: {awake_count} | Asleep: {asleep_count}\n\n" + "\n\n".join(details_list)
        messagebox.showinfo("Detection Summary", summary_msg)
    else:
        messagebox.showinfo("No Faces Found", "No faces detected in the selected image.")

# === GUI Setup ===
root = tk.Tk()
root.title("Drowsiness & Age Detection")

btn = tk.Button(root, text="Select Image", command=select_image,
                font=("Arial", 14), bg="#4CAF50", fg="white", padx=12, pady=6)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()
