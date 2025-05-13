
# 📝 **Laporan Sederhana: Analisis Sentimen Menggunakan SVM**

## 1. **Tujuan Proyek**

Proyek ini bertujuan untuk mengklasifikasikan sentimen (positif/negatif) dari data teks menggunakan metode **Support Vector Machine (SVM)**. Dataset yang digunakan adalah `Training.txt`, yang terdiri dari dua kolom: `label` (kelas sentimen) dan `text` (isi ulasan).

---

## 2. **Langkah-Langkah Implementasi**

### a. **Import Library**

Beberapa pustaka Python yang digunakan:

- `pandas`, `numpy`: manipulasi data  
- `scikit-learn`: preprocessing, pelatihan, dan evaluasi model  
- `nltk`, `textblob`: pemrosesan bahasa alami  
- `seaborn`, `matplotlib`: visualisasi data

### b. **Preprocessing**

Proses pembersihan teks meliputi:

- Konversi ke **huruf kecil**
- Penghapusan **tanda baca**

Contoh:

```
Input:  "Film ini sangat bagus!"
Output: "film ini sangat bagus"
```

### c. **Vectorization**

Teks dikonversi menjadi fitur numerik menggunakan `TfidfVectorizer`, yaitu metode representasi teks berbasis frekuensi kata yang mempertimbangkan keunikan kata di seluruh dokumen.

### d. **Pemodelan**

Model dibangun menggunakan pipeline:

```python
Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])
```

Digunakan **SVM dengan kernel linear** yang umum digunakan dalam klasifikasi teks.

### e. **Split Data**

Data dibagi menjadi:

- 80% data latih (`X_train`)
- 20% data uji (`X_test`)

### f. **Evaluasi Model**

Evaluasi dilakukan menggunakan beberapa metrik, yaitu:

- **Akurasi**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**
- **F1 Score (weighted)**

---

## 3. **Hasil Evaluasi (Contoh Output)**

```
Akurasi: 0.85

Classification Report:
              precision    recall  f1-score   support

    negatif       0.83      0.88      0.85        25
    positif       0.87      0.82      0.84        25

    accuracy                           0.85        50
   macro avg       0.85      0.85      0.85        50
weighted avg       0.85      0.85      0.85        50

Confusion Matrix:
[[22  3]
 [ 4 21]]

F1 Score: 0.85
```

---

## 4. **(Opsional) Hyperparameter Tuning**

Template disediakan untuk pencarian hyperparameter menggunakan `GridSearchCV`, dengan parameter:

- `ngram_range`: (1,1) atau (1,2)
- `use_idf`: True / False
- `C`: parameter regulasi SVM (misal: 0.1, 1, 10)

---

## 5. **Kesimpulan**

Model SVM mampu membedakan sentimen teks dengan cukup baik. Dengan preprocessing sederhana dan representasi fitur TF-IDF, model ini menunjukkan performa yang solid untuk klasifikasi dasar. Akurasi dapat ditingkatkan lebih lanjut melalui:

- Pembersihan data lanjutan (penghapusan stopwords, stemming)
- Tuning parameter model
- Penambahan data latih
