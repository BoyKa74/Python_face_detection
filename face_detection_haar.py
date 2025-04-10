import cv2
from argparse import ArgumentParser

from utils import haar_face_to_points


# trích xuất đường dẫn ảnh (tham số khi chạy code)
parser = ArgumentParser(description="Apply Haar Cascade model on images")
parser.add_argument('--image', dest="image_path", help='Path to image', required=True)
args = parser.parse_args()
image_path = args.image_path

# load ảnh và chuyển sang ảnh đen trắng 
# (mô hình cần ảnh đen trắng đầu vào)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Load ảnh: {}".format(image_path))
print("Cỡ ảnh: {}".format(gray.shape))

# load mô hình nhận diện khuôn mặt
model_path = "models/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(model_path)

# nhận diện khuôn mặt trong ảnh
faces = detector.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=5)
print("Xác định được {} khuôn mặt".format(len(faces)))

# Vẽ đường bao cho từng khuôn mặt
green_color = (0, 255, 0)
for face in faces:
    x1, y1, x2, y2 = haar_face_to_points(face)
    cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=green_color, thickness=2)

# Hiển thị ảnh
cv2.imshow("Haar Cascade", image)
cv2.waitKey(0)