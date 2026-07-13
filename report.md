**Tugas Praktikum: Fine-tuning BERT/IndoBERT untuk Klasifikasi Teks Sederhana**

**1. Judul praktikum.**  
Fine-tuning IndoBERT untuk Klasifikasi Teks Sederhana

**2. Deskripsi dataset.**  
Dataset berisi ulasan produk dalam bahasa Indonesia dengan label sentimen positif dan negatif.

**3. Jumlah data dan label.**  
- Total data: 60  
- Label: positif (30), negatif (30)

**4. Tahapan preprocessing/tokenization.**  
- Menggunakan Hugging Face AutoTokenizer dari model indobenchmark/indobert-base-p1  
- Max length: 128, truncation dan padding

**5. Model yang digunakan.**  
indobenchmark/indobert-base-p1 (IndoBERT)

**6. Hasil training.**  
Dilakukan fine-tuning selama 1 epoch dengan batch size 8.

**7. Hasil evaluasi.**  
- Accuracy: 0.9167  
- F1-score: 0.9167

**8. Kesimpulan.**  
Model IndoBERT berhasil melakukan klasifikasi teks sederhana dengan baik setelah fine-tuning minimal 1 epoch.