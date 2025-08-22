import cv2
import os

# Memuat model deteksi wajah dari file XML (Haar Cascade)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Memuat model pengenalan wajah yang sudah dilatih
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Memuat mapping ID ke nama dari file teks
id_map = {}
try:
    with open('id_map.txt', 'r') as f:
        for line in f:
            user_id, name = line.strip().split(':')
            id_map[int(user_id)] = name
except FileNotFoundError:
    print("[ERROR] File 'id_map.txt' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
    exit()

# Inisialisasi kamera dengan resolusi yang lebih tinggi
camera = cv2.VideoCapture(0)
# Mengatur lebar dan tinggi frame untuk kualitas yang lebih baik
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n[INFO] Program pengenalan wajah siap. Arahkan wajahmu ke kamera.")
print("[INFO] Tekan 'Q' untuk keluar.")

while True:
    success, frame = camera.read()
    if not success:
        print("Gagal mengambil frame dari kamera.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Menggambar kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Melakukan prediksi menggunakan model
        user_id, confidence = recognizer.predict(gray_frame[y:y + h, x:x + w])

        # Menggunakan confidence (tingkat kepercayaan) untuk menentukan kecocokan
        # Ambang batas 100 menandakan ketidakcocokan
        if confidence < 100:
            name = id_map.get(user_id, "Tidak Dikenal")
        else:
            name = "Tidak Dikenal"

        # Menampilkan nama saja di layar, tanpa confidence
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("\n[INFO] Program selesai.")