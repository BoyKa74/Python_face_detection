# Import các thư viện cần thiết
import os
import cv2
import dlib
import numpy as np
from flask import Flask, request, send_file, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from user_manager import UserManager

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Khóa bí mật cho session
UPLOAD_FOLDER = "uploads"  # Thư mục lưu ảnh tải lên
RESULT_FOLDER = "results"  # Thư mục lưu ảnh kết quả

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load các mô hình nhận diện khuôn mặt
haar_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")  # Mô hình Haar Cascade
hog_detector = dlib.get_frontal_face_detector()  # Mô hình HOG
dnn_net = cv2.dnn.readNetFromTensorflow(  # Mô hình DNN
    "models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt"
)

# Khởi tạo UserManager để quản lý người dùng
user_manager = UserManager()

def detect_faces(image_path, method):
    """
    Hàm nhận diện khuôn mặt trong ảnh
    Args:
        image_path: Đường dẫn đến ảnh cần xử lý
        method: Phương pháp nhận diện (haar, hog, dnn)
    Returns:
        Đường dẫn đến ảnh kết quả đã vẽ khung khuôn mặt
    """
    # Đọc ảnh và chuyển sang grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt theo phương pháp được chọn
    if method == "haar":
        faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
    elif method == "hog":
        faces = hog_detector(gray)
        faces = [(d.left(), d.top(), d.width(), d.height()) for d in faces]
    elif method == "dnn":
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        faces = []
        h, w = image.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2 - x1, y2 - y1))

    # Vẽ khung xanh lá cây quanh khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Lưu ảnh kết quả
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, image)
    return result_path

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
    if not user_manager.is_logged_in():
        return "Unauthorized", 401
        
    if "image" not in request.files:
        return "Không có ảnh được tải lên", 400
    file = request.files["image"]
    method = request.form.get("method")

    if file.filename == "":
        return "Không có file được chọn", 400

    # Lưu file tải lên
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Xử lý ảnh và trả về kết quả
    result_path = detect_faces(file_path, method)
    return send_file(result_path, mimetype="image/jpeg")

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

if __name__ == "__main__":
    app.run(debug=True)
