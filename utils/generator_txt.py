import datetime
from pathlib import Path

def generate_txt(image_path, prediction, confidence, nama_pasien, usia, jenis_kelamin, output_path="diagnosa_kulit.txt"):
    try:
        output_file = Path(output_path)
        with output_file.open("w", encoding="utf-8") as f:
            f.write("HASIL DIAGNOSA DETEKSI PENYAKIT KULIT - AI MODEL (VGG16)\n")
            f.write("=========================================================\n")
            f.write(f"Tanggal Pemeriksaan: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n")

            f.write("DATA PASIEN:\n")
            f.write(f"- Nama Pasien     : {nama_pasien}\n")
            f.write(f"- Usia            : {usia} tahun\n")
            f.write(f"- Jenis Kelamin   : {jenis_kelamin}\n\n")

            f.write("HASIL PREDIKSI:\n")
            f.write(f"- Jenis Penyakit   : {prediction}\n")
            f.write(f"- Tingkat Keyakinan: {confidence:.2f}%\n\n")

            if image_path:
                f.write(f"- Gambar Asal      : {image_path}\n\n")

            f.write("CATATAN:\n")
            pred = (prediction or "").strip().lower()
            if pred == "melanoma":
                f.write("Melanoma adalah kanker kulit ganas yang dapat membahayakan jiwa.\n")
                f.write("Segera konsultasikan dengan dokter spesialis kulit untuk pemeriksaan lebih lanjut.\n")
            elif pred == "psoriasis":
                f.write("Psoriasis adalah inflamasi kulit kronis yang tidak menular namun dapat kambuh.\n")
                f.write("Dianjurkan untuk menjaga kelembaban kulit dan menghindari stres berlebih.\n")
            else:
                f.write("Jenis penyakit tidak diketahui atau tidak dikenali oleh sistem.\n")

            f.write("\nDisclaimer:\n")
            f.write("Informasi ini bukan pengganti diagnosa medis resmi. Selalu konsultasikan dengan profesional kesehatan.\n")
        
        return str(output_file)
    
    except Exception as e:
        raise RuntimeError(f"Gagal menulis laporan diagnosa: {e}")
