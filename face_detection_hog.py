import cv2
import dlib
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load mô hình nhận diện khuôn mặt
hog_face_detector = dlib.get_frontal_face_detector()

def hog_face_to_points(face):
    """Chuyển đổi đối tượng khuôn mặt của dlib thành tọa độ"""
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    return x1, y1, x2, y2

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file.stream)
    open_cv_image = np.array(img.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt
    faces = hog_face_detector(gray)

    print(f"Detected {len(faces)} faces")

    # Vẽ hình chữ nhật quanh khuôn mặt
    green_color = (0, 255, 0)
    for face in faces:
        x1, y1, x2, y2 = hog_face_to_points(face)
        cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), green_color, 2)

    # Chuyển ảnh về dạng có thể hiển thị trên web
    _, img_encoded = cv2.imencode('.jpg', open_cv_image)
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
