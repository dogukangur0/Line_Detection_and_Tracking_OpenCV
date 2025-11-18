
# 🛣️🚦 Line Detection & Tracking with OpenCV

> Bu proje, yol şeritlerini tespit etmek ve ardışık video kareleri boyunca bu şeritleri takip etmek için OpenCV tabanlı bir görüntü işleme pipeline'ı içerir. 
> Sistem, renk uzayı dönüşümleri, ROI seçimi, Canny edge detection, HoughLinesP ve geometrik analiz kullanarak stabil şerit tespiti sağlar.

### 📌 Özellikler
* ✔️ BGR → HSV renk dönüşümü
* ✔️ Sarı ve beyaz şeritler için renk bazlı maskeleme
* ✔️ Gürültüyü azaltmak için Gaussian Blur
* ✔️ ROI (Region of Interest) kullanarak gereksiz bölgeleri filtreleme
* ✔️ Canny edge detection
* ✔️ Probabilistic HoughLinesP ile çizgi tespiti

### ⚙️ Kullanılan Teknolojiler
- Python
- OpenCV
- NumPy

[Video Dataset](https://www.kaggle.com/datasets/ashokkumarindia/road-lane-line-detection-project)
