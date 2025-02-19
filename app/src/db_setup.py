# API/app/src/check_db.py
from pymongo import MongoClient
import sys
import os
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def check_database():
    try:
        # Try to connect
        print("Mencoba koneksi ke MongoDB...")
        client = MongoClient(Config.MONGODB_URI)
        
        # Check if server is available
        client.server_info()
        print("✓ Koneksi ke MongoDB berhasil")
        
        # Check database
        db = client[Config.DB_NAME]
        print(f"\nMemeriksa database '{Config.DB_NAME}'...")
        print(f"Collections yang tersedia: {db.list_collection_names()}")
        
        # Check mahasiswas collection
        print("\nMemeriksa collection 'mahasiswas'...")
        count = db.mahasiswas.count_documents({})
        print(f"Jumlah dokumen: {count}")
        
        if count > 0:
            print("\nContoh data pertama:")
            pprint(db.mahasiswas.find_one())
        else:
            print("\nMenambahkan data sample...")
            sample_data = [
                {
            "ID_SISWA": "38606",
            "NO_DAFTAR": "2212012488",
            "NPM": "2209010298",
            "NAMA_LENGKAP": "RAYHAN PRATAMA",
            "JENIS_KELAMIN": "Laki-Laki",
            "TGL_LAHIR": "2004-08-20 00:00:00.000",
            "TEMPAT_LAHIR": "MEDAN",
            "ALAMAT": "----",
            "PROVINSI": "----",
            "KABKOTA": "----",
            "KELAS": "A1 Siang",
            "ANGKATAN": 2022,
            "FAKULTAS": "Ilmu Komputer dan Teknologi Informasi",
            "ID_PRODI": 86,
            "SINGKATAN_PRODI": "SI",
            "PRODI": "Sistem Informasi",
            "EMAIL": "agarayhan123@gmail.com",
            "TAHUN_MASUK": 2022,
            "ASAL_SEKOLAH": "----",
            "PROVINSI_SEKOLAH": "----",
            "KABKOTA_SEKOLAH": "----",
            "STATUS_AKTIF": "Aktif",
            "TGL_MASUK": "2022-10-02 18:29:49.000",
            "TGL_LULUS": "----"
                },
                {
            "ID_SISWA": "38607",
            "NO_DAFTAR": "2212012489",
            "NPM": "2209010297",
            "NAMA_LENGKAP": "MUHAMMAD RIAN MIZARD",
            "JENIS_KELAMIN": "Laki-Laki",
            "TGL_LAHIR": "2002-06-25 00:00:00.000",
            "TEMPAT_LAHIR": "Belawan",
            "ALAMAT": "----",
            "PROVINSI": "----",
            "KABKOTA": "----",
            "KELAS": "A1 Siang",
            "ANGKATAN": 2022,
            "FAKULTAS": "Ilmu Komputer dan Teknologi Informasi",
            "ID_PRODI": 86,
            "SINGKATAN_PRODI": "SI",
            "PRODI": "Sistem Informasi",
            "EMAIL": "muhammadrianmizard@gmail.com",
            "TAHUN_MASUK": 2022,
            "ASAL_SEKOLAH": "----",
            "PROVINSI_SEKOLAH": "----",
            "KABKOTA_SEKOLAH": "----",
            "STATUS_AKTIF": "Aktif",
            "TGL_MASUK": "2022-10-02 18:29:49.000",
            "TGL_LULUS": "----"
                }
            ]
            
            result = db.mahasiswas.insert_many(sample_data)
            print(f"✓ Berhasil menambahkan {len(result.inserted_ids)} data sample")
            
            print("\nContoh data yang baru ditambahkan:")
            pprint(db.mahasiswas.find_one())
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nKemungkinan penyebab:")
        print("1. MongoDB belum berjalan")
        print("2. Port 27017 tidak tersedia")
        print("3. Database memerlukan autentikasi")
        print("\nSilakan periksa:")
        print("1. Apakah MongoDB sudah diinstall?")
        print("2. Apakah service MongoDB sudah berjalan?")
        print("3. Coba buka MongoDB Compass dan koneksikan ke localhost:27017")

if __name__ == "__main__":
    check_database()