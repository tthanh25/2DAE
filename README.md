# Phát hiện và phòng thủ chống lại các mẫu đối kháng trong học sâu sử dụng thống kê cảnh tự nhiên và khử nhiễu thích ứng.​
​link google colab: https://colab.research.google.com/drive/1TFcNLOb31KJ_gqkUrX_qNgBrM3bys1-s#scrollTo=Q4SaDeY1PLhZ

## Các thư viện cần thiết:
  ```
  !pip install matplotlib cleverhans pandas numpy keras scipy Pillow scikit-image imageio joblib scikit-learn
  !pip install tensorflow
  !pip install bm3d
  ```
Phiên bản: 
TensorFlow version: 2.17.0 ​
CleverHans version: 4.0.0​
bm3d version: 4.0.3

Trong quá trình cài đặt thư viện nếu báo lỗi thiếu thì các bạn import thêm nhé
## Thứ tự chạy file: 
  1. MNIST_model.py
  2. data_preparation_denoiser.py và data_preparation_detector.py
  3. train_detector.py và train_denoiser.py
  4. chạy file test như test.py / testDenoiser.py / testDetector.py
## P/s:
    - mô hình train với bộ dữ liệu MNIST
    - mô hình denoiser được train với khoảng 1650 ảnh nên có độ chính xác không cao đối với hình ảnh bị làm nhiễu mạnh.
    - mình đã train và có dữ liệu sẵn nên các bạn muốn trải nghiệm thì chỉ cần chạy file test. Nếu không hãy xóa những file có đuôi .h5 / .npz / .joblib đi nhé
