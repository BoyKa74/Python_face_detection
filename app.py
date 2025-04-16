# Import các thư viện cần thiết
import os
import cv2
import dlib
import numpy as np
from flask import Flask, request, send_file, render_template, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from user_manager import UserManager
import base64
import shutil
from datetime import datetime

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Khóa bí mật cho session
UPLOAD_FOLDER = "uploads"  # Thư mục lưu ảnh tải lên
RESULT_FOLDER = "results"  # Thư mục lưu ảnh kết quả
STATIC_FOLDER = "static/results"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load các mô hình nhận diện khuôn mặt
haar_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")  # Mô hình Haar Cascade
hog_detector = dlib.get_frontal_face_detector()  # Mô hình HOG
dnn_net = cv2.dnn.readNetFromTensorflow(  # Mô hình DNN
    "models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt"
)

# Load mô hình nhận diện giới tính
gender_net = cv2.dnn.readNet(
    'models/gender_net.caffemodel',
    'models/gender_deploy.prototxt'
)

# Labels cho giới tính
GENDER_LIST = ['Nam', 'Nữ']

# Khởi tạo UserManager để quản lý người dùng
user_manager = UserManager()

def detect_faces_and_gender(image_path, method):
    """
    Hàm nhận diện khuôn mặt và giới tính trong ảnh
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Không thể đọc ảnh")

    # Chuyển sang grayscale cho nhận diện khuôn mặt
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = []
    # Nhận diện khuôn mặt theo phương pháp được chọn
    if method == "haar":
        face_rects = haar_cascade.detectMultiScale(gray, 1.1, 4)
        faces = [(x, y, x+w, y+h) for (x, y, w, h) in face_rects]
    elif method == "hog":
        face_rects = hog_detector(gray)
        faces = [(f.left(), f.top(), f.right(), f.bottom()) for f in face_rects]
    else:  # dnn
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                faces.append(tuple(box.astype("int")))

    # Thống kê
    stats = {
        'total': len(faces),
        'Nam': 0,
        'Nữ': 0
    }

    # Xử lý từng khuôn mặt
    for (x1, y1, x2, y2) in faces:
        try:
            # Cắt và chuẩn bị ảnh khuôn mặt
            face_img = image[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Resize ảnh cho mô hình giới tính
            face_blob = cv2.dnn.blobFromImage(
                cv2.resize(face_img, (227, 227)),
                1.0,
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            # Dự đoán giới tính
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            confidence = gender_preds[0].max() * 100

            # Cập nhật thống kê
            stats[gender] += 1

            # Vẽ khung và nhãn
            color = (0, 255, 0) if gender == "Nam" else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{gender} ({confidence:.1f}%)"
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print(f"Lỗi khi nhận diện giới tính: {str(e)}")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Lưu ảnh kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"result_{timestamp}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, image)

    return result_path, result_filename, stats

# Route cho trang chủ
@app.route("/")
def index():
    if not user_manager.is_logged_in():
        return redirect(url_for('login'))
    return redirect(url_for('home'))

# Route cho trang home
@app.route("/home")
def home():
    if not user_manager.is_logged_in():
        return redirect(url_for('login'))
    return render_template("home.html", username=session['username'])

# Route cho trang detection
@app.route("/detection")
def detection():
    if not user_manager.is_logged_in():
        return redirect(url_for('login'))
    return render_template("index.html", username=session['username'])

# Route cho trang đăng nhập
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        success, message = user_manager.login(username, password)
        if success:
            return redirect(url_for('home'))
        return render_template("login.html", error=message)
    return render_template("login.html")

# Route cho trang đăng ký
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if password != confirm_password:
            return render_template("register.html", error="Mật khẩu không khớp")
        
        success, message = user_manager.register(username, password)
        if success:
            return redirect(url_for('login'))
        return render_template("register.html", error=message)
    return render_template("register.html")

# Route cho đăng xuất
@app.route("/logout")
def logout():
    user_manager.logout()
    return redirect(url_for('login'))

# Route cho việc tải ảnh lên và xử lý
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Không có ảnh được tải lên"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Không có file được chọn"}), 400

        # Lưu file tải lên
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Xử lý ảnh
        method = request.form.get("method", "haar")
        result_path, result_filename, stats = detect_faces_and_gender(file_path, method)

        # Đọc ảnh kết quả và chuyển về base64
        with open(result_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Trả về kết quả
        return jsonify({
            'success': True,
            'image': image_data,
            'filename': result_filename,
            'stats': stats
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Route cho việc tải xuống ảnh kết quả
@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

# Route cho trang giới thiệu
@app.route("/about")
def about():
    return render_template("about.html")

# Route cho trang FAQ
@app.route("/faq")
def faq():
    return render_template("faq.html")

# Route cho trang liên hệ
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # Xử lý form liên hệ
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")
        # TODO: Thêm logic xử lý form liên hệ
        flash("Cảm ơn bạn đã liên hệ với chúng tôi!", "success")
        return redirect(url_for('contact'))
    return render_template("contact.html")

@app.route("/save_image", methods=["POST"])
def save_image():
    try:
        data = request.json
        filename = data.get('filename')
        if not filename:
            return jsonify({"success": False, "error": "Không có tên file"}), 400

        # Copy ảnh từ thư mục tạm sang static
        src_path = os.path.join(RESULT_FOLDER, filename)
        dst_path = os.path.join(STATIC_FOLDER, filename)
        shutil.copy2(src_path, dst_path)

        return jsonify({
            'success': True,
            'message': 'Đã lưu ảnh thành công',
            'path': f'/static/results/{filename}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
