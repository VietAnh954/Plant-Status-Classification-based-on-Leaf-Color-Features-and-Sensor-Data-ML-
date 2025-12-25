# Plant-Status-Classification-based-on-Leaf-Color-Features-and-Sensor-Data-ML-
ỨNG DỤNG HỌC MÁY TRONG PHÂN LOẠI TRẠNG THÁI CÂY TRỒNG DỰA TRÊN ĐẶC TRƯNG MÀU SẮC LÁ VÀ DỮ LIỆU CẢM BIẾN
# 🌿 Ứng dụng Học máy Phân loại Sức khỏe Cây trồng (Multimodal Plant Health Diagnosis)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV-green)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

> **Đề tài:** Ứng dụng Học máy trong phân loại trạng thái cây trồng dựa trên đặc trưng màu sắc lá và dữ liệu cảm biến môi trường.

## 📖 Tổng quan (Overview)

Dự án này xây dựng một hệ thống chẩn đoán sức khỏe cây trồng thông minh, giải quyết bài toán nông nghiệp công nghệ cao bằng cách kết hợp hai nguồn dữ liệu:
1.  **Thị giác máy tính (Computer Vision):** Trích xuất đặc trưng màu sắc từ ảnh lá cây.
2.  **Internet of Things (IoT):** Dữ liệu cảm biến môi trường (Độ ẩm đất, Ánh sáng).

Hệ thống có khả năng phân loại 4 trạng thái của cây:
* ✅ **Tốt (Healthy):** Cây phát triển bình thường.
* 💧 **Cần tưới (Needs Water):** Cây thiếu nước nhưng chưa héo rũ.
* 🍂 **Héo (Wilted):** Cây bị stress nặng do thiếu nước và nắng gắt.
* 🦠 **Sâu bệnh (Diseased):** Cây bị tấn công bởi vi khuẩn, nấm hoặc virus.

---

## 🏗️ Kiến trúc hệ thống (Workflow)

Dự án được chia thành 5 giai đoạn xử lý (tương ứng với 5 Notebooks):

1.  **Data Loading:** Tải và khám phá dữ liệu ảnh thô.
2.  **Feature Extraction:** Biến đổi ảnh thành dữ liệu số (Tỷ lệ màu sắc).
3.  **Sensor Processing:** Làm sạch và chuẩn hóa dữ liệu cảm biến.
4.  **Data Hybridization:** Lai tạo dữ liệu ảnh và cảm biến dựa trên luật sinh học (Simulation).
5.  **Modeling:** Huấn luyện và đánh giá mô hình Random Forest.

---

## 📂 Chi tiết thực hiện (Project Details)

### 1. [`1_Load_PlantVillageData_Visualization.ipynb`](./1_Load_PlantVillageData_Visualization.ipynb)
* **Mục tiêu:** Thu thập dữ liệu hình ảnh từ bộ dữ liệu chuẩn **PlantVillage**.
* **Kỹ thuật:**
    * Sử dụng `kagglehub` để tải dữ liệu tự động.
    * Tổ chức thư mục: `Raw_Data` -> `Processed_Data`.
* **Phát hiện quan trọng:** Phân tích biểu đồ phân phối cho thấy sự **mất cân bằng dữ liệu (Class Imbalance)** lớn: Ảnh cây bệnh chiếm đa số so với ảnh cây khỏe.

### 2. [`02_Feature_Extraction.ipynb`](./02_Feature_Extraction.ipynb)
* **Mục tiêu:** Trích xuất đặc trưng định lượng từ ảnh (Feature Engineering). Thay vì dùng CNN nặng nề, dự án sử dụng phương pháp xử lý ảnh cơ bản để tối ưu hiệu năng.
* **Kỹ thuật:**
    * Chuyển đổi không gian màu: **RGB -> HSV** (Tách biệt thông tin màu sắc và độ sáng).
    * **Color Masking:** Tạo các mặt nạ để tách màu Xanh (Lá khỏe), Vàng (Lá già/Héo), Nâu (Hoại tử/Bệnh).
    * Tính toán tỷ lệ phần trăm pixel (`Pct_Green`, `Pct_Yellow`, `Pct_Brown`) trên diện tích lá.
* **Kết quả:** File `leaf_features_final.csv` chứa thông tin màu sắc của hàng nghìn bức ảnh.

### 3. [`3_Sensor_Data_Processing.ipynb`](./3_Sensor_Data_Processing.ipynb)
* **Mục tiêu:** Xử lý dữ liệu môi trường từ 2 nguồn dữ liệu mở (Smart Farming & Crop Recommendation).
* **Kỹ thuật:**
    * Đồng nhất tên cột (`Soil_Moisture`, `Sunlight_Hours`).
    * **Imputation:** Xử lý dữ liệu thiếu bằng logic nghiệp vụ (Thiếu nhãn -> Giả định là Healthy).
    * **Feature Engineering:** Chuyển đổi chỉ số `Pest Pressure` (Áp lực sâu bệnh) thành nhãn phân loại (Ngưỡng cắt > 70).
* **Kết quả:** 2 file dữ liệu cảm biến sạch đã sẵn sàng để ghép nối.

### 4. [`4_Final_Dataset_Creation.ipynb`](./4_Final_Dataset_Creation.ipynb) ⭐ *(Trọng tâm)*
* **Mục tiêu:** Giải quyết vấn đề thiếu hụt dữ liệu đồng bộ bằng kỹ thuật **Lai tạo dữ liệu (Data Hybridization)**.
* **Kỹ thuật:** Xây dựng hàm mô phỏng `hybridize_row` dựa trên luật chuyên gia (Expert Rules):
    * `Lá Xanh` + `Đất Ẩm` = **Tốt**.
    * `Lá Xanh` + `Đất Hơi Khô` = **Cần Tưới**.
    * `Lá Vàng` + `Đất Rất Khô` = **Héo**.
    * `Lá Nâu/Đốm` + `Áp lực bệnh cao` = **Sâu Bệnh**.
* **Kết quả:** File `Final_Training_Data.csv` (Dataset tổng hợp đa phương thức).

### 5. [`5_Model_Training_Evaluation.ipynb`](./5_Model_Training_Evaluation.ipynb)
* **Mục tiêu:** Huấn luyện mô hình AI phân loại.
* **Thuật toán:** **Random Forest Classifier** (`n_estimators=100`).
* **Kỹ thuật:**
    * `Stratified Split`: Chia tập train/test đảm bảo tỷ lệ nhãn.
    * **Stress Test:** Thêm nhiễu Gaussian vào dữ liệu test để kiểm tra độ bền vững (Robustness) của mô hình trước sai số cảm biến thực tế.
    * Phân tích **Feature Importance** và **Confusion Matrix**.

---

## 📊 Kết quả (Results)

| Metric | Giá trị | Nhận xét |
| :--- | :--- | :--- |
| **Accuracy** | **90.39%** | Độ chính xác cao trên tập kiểm thử. |
| **Precision (Healthy)** | **1.00** | Tuyệt đối chính xác khi nhận diện cây khỏe. |
| **Recall (Disease)** | **0.95** | Rất nhạy trong việc phát hiện sâu bệnh (Ưu tiên an toàn mùa màng). |
| **Stress Test** | **~72%** | Mô hình vẫn hoạt động tốt (không bị crash) khi dữ liệu đầu vào bị nhiễu nặng. |

---

## 🚀 Hướng dẫn cài đặt (Installation)

1.  Clone repository này về máy:
    ```bash
    git clone [https://github.com/USERNAME/SmartPlant-Disease-Diagnosis-AI.git](https://github.com/USERNAME/SmartPlant-Disease-Diagnosis-AI.git)
    ```
2.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install pandas numpy opencv-python scikit-learn seaborn matplotlib
    ```
3.  Chạy lần lượt các Notebook từ 1 đến 5 để tái hiện quá trình xử lý và huấn luyện.
4.  Mô hình đã huấn luyện được lưu tại: `plant_health_rf_model.joblib`.

---

## 🔮 Hướng phát triển (Future Work)
* Tích hợp kỹ thuật **SMOTE** để cân bằng lại dữ liệu cho lớp 'Héo' (Wilted).
* Triển khai mô hình lên thiết bị **IoT Edge (Raspberry Pi)** để chạy thời gian thực.
* Phát triển Module tự động tách nền (Background Removal) để xử lý ảnh chụp thực tế tốt hơn.

---

## 👨‍💻 Tác giả (Author)
* **Name:** [NGUYEN VIET ANH]
* **University:** [PTITHCM]
* **Project:** Đồ án môn học Machine Learning.

---
*If you find this project useful, please give it a star ⭐!*
