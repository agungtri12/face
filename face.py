import cv2
import os

# Memuat model deteksi wajah dari file XML (Haar Cascade)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Meminta pengguna untuk memasukkan nama
user_name = input("Masukkan nama Anda (gunakan huruf kecil, tanpa spasi): ")
if not user_name.isalpha():
    print("Nama tidak valid. Harap gunakan hanya huruf.")
    exit()

# Tentukan direktori untuk menyimpan gambar
output_dir = os.path.join('dataset_wajah', user_name)
image_count = 0

# =========================================================================
# LOGIKA UNTUK MENGECEK DAN MENAMBAHKAN DATA KE FOLDER YANG ADA
# =========================================================================
# Periksa apakah folder dengan nama yang dimasukkan sudah ada
if os.path.exists(output_dir):
    print(f"\nFolder '{output_dir}' sudah ada.")
    choice = input("Apakah Anda ingin menambahkan foto ke folder ini? (ya/tidak): ").lower()
    
    if choice == 'ya':
        # Jika pengguna memilih untuk menambahkan, cari nomor file gambar terakhir
        files = [f for f in os.listdir(output_dir) if f.startswith(f"{user_name}_") and f.endswith(".jpg")]
        if files:
            # Dapatkan nomor tertinggi dari nama file dan lanjutkan dari sana
            image_numbers = [int(f.split('_')[-1].split('.')[0]) for f in files]
            image_count = max(image_numbers) + 1
        print(f"Melanjutkan pengambilan gambar. Mulai dari gambar ke-{image_count}.")
    else:
        # Jika pengguna memilih tidak, program akan keluar
        print("Pengambilan data dibatalkan.")
        exit()
else:
    # Jika folder tidak ada, buat folder baru
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' berhasil dibuat.")
# =========================================================================

# Inisialisasi kamera
camera = cv2.VideoCapture(0)

# =========================================================================
# MENYESUAIKAN KUALITAS KAMERA
# =========================================================================
# Mengatur lebar, tinggi, kecerahan, dan kontras untuk kualitas gambar yang optimal
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_BRIGHTNESS, 100)
camera.set(cv2.CAP_PROP_CONTRAST, 50)
# =========================================================================

# Inisialisasi status deteksi wajah
found_face = False

print(f"\nProgram siap mengambil gambar untuk {user_name}. Arahkan wajahmu ke kamera.")
print("Tekan tombol 'S' untuk menyimpan gambar, dan 'Q' untuk keluar.")

while True:
    success, frame = camera.read()
    if not success:
        print("Gagal mengambil frame dari kamera.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Memeriksa apakah wajah terdeteksi
    if len(faces) > 0:
        found_face = True
    else:
        found_face = False

    for (x, y, w, h) in faces:
        # Menggambar kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    # Menampilkan umpan balik visual kepada pengguna
    info_text = f"Gambar diambil: {image_count}"
    if found_face:
        cv2.putText(frame, "Wajah Terdeteksi!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Arahkan wajahmu ke kamera", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Face Data Capture', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_crop = frame[y:y + h, x:x + w]
        
        # Menyimpan gambar dengan kualitas tertinggi (100)
        image_filename = os.path.join(output_dir, f'{user_name}_{image_count}.jpg')
        cv2.imwrite(image_filename, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        print(f"Gambar ke-{image_count} berhasil disimpan: {image_filename}")
        image_count += 1
        
    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print(f"\nProses selesai. Total {image_count} gambar wajah untuk {user_name} telah disimpan.")
