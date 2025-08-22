import cv2
import os
import numpy as np
from PIL import Image

# Path ke direktori utama dataset
dataset_path = "dataset_wajah"

# Inisialisasi recognizer (kita akan menggunakan LBPHFaceRecognizer)
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    """
    Fungsi untuk mengambil gambar dan label dari direktori dataset yang terstruktur
    berdasarkan nama sub-folder.
    """
    image_paths = []
    face_samples = []
    ids = []
    
    # Mapping nama folder ke ID numerik
    # Misalnya: {"agung": 0, "budi": 1}
    id_map = {}
    current_id = 0

    # Dapatkan nama-nama folder (nama orang) di dalam direktori dataset
    user_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    
    # Buat mapping ID numerik untuk setiap nama orang
    for user_name in user_names:
        if user_name not in id_map:
            id_map[user_name] = current_id
            current_id += 1
            print(f"[INFO] Menemukan orang: {user_name} dengan ID: {id_map[user_name]}")

    # Loop melalui setiap folder (setiap orang)
    for user_name in user_names:
        user_path = os.path.join(path, user_name)
        # Loop melalui setiap gambar di dalam folder orang tersebut
        for image_file in os.listdir(user_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(user_path, image_file)
                pil_image = Image.open(image_path).convert('L')
                img_numpy = np.array(pil_image, 'uint8')

                # Ambil ID yang sesuai dari mapping
                img_id = id_map[user_name]
                
                face_samples.append(img_numpy)
                ids.append(img_id)
                image_paths.append(image_path)
    
    return face_samples, ids, id_map

print("\n[INFO] Memulai pelatihan model pengenalan wajah. Mohon tunggu...")

# Ambil gambar dan label dari dataset
faces, ids, id_map = get_images_and_labels(dataset_path)

if len(faces) > 0:
    # Melatih model menggunakan data yang sudah disiapkan
    recognizer.train(faces, np.array(ids))

    # Simpan model yang sudah dilatih ke file
    model_filename = 'trainer.yml'
    recognizer.write(model_filename)
    
    # Simpan juga mapping ID ke nama untuk program pengenalan
    with open('id_map.txt', 'w') as f:
        for name, user_id in id_map.items():
            f.write(f"{user_id}:{name}\n")
    
    print(f"\n[INFO] Model pengenalan wajah berhasil dilatih dan disimpan sebagai '{model_filename}'")
    print(f"[INFO] Total {len(faces)} gambar dari {len(id_map)} orang dilatih.")
    print(f"[INFO] Mapping ID-Nama disimpan ke 'id_map.txt'")
else:
    print("\n[INFO] Tidak ada gambar yang ditemukan untuk dilatih. Pastikan folder 'dataset_wajah' memiliki sub-folder dengan gambar.")