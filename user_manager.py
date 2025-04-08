# Import các thư viện cần thiết
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session

class UserManager:
    """
    Lớp quản lý người dùng, xử lý đăng ký, đăng nhập và đăng xuất
    """
    def __init__(self):
        # Khởi tạo đường dẫn đến file lưu thông tin người dùng
        self.users_file = "users.json"
        # Tạo file users.json nếu chưa tồn tại
        if not os.path.exists(self.users_file):
            with open(self.users_file, "w") as f:
                json.dump({}, f)
        # Đọc thông tin người dùng từ file
        self.users = self._load_users()

    def _load_users(self):
        """
        Đọc thông tin người dùng từ file JSON
        Returns:
            dict: Từ điển chứa thông tin người dùng
        """
        with open(self.users_file, "r") as f:
            return json.load(f)

    def _save_users(self):
        """
        Lưu thông tin người dùng vào file JSON
        """
        with open(self.users_file, "w") as f:
            json.dump(self.users, f)

    def register(self, username, password):
        """
        Đăng ký người dùng mới
        Args:
            username: Tên đăng nhập
            password: Mật khẩu
        Returns:
            tuple: (success, message) - Trạng thái đăng ký và thông báo
        """
        # Kiểm tra tên đăng nhập đã tồn tại chưa
        if username in self.users:
            return False, "Tên đăng nhập đã tồn tại"
        
        # Thêm người dùng mới
        self.users[username] = {
            'password': generate_password_hash(password)
        }
        self._save_users()
        return True, "Đăng ký thành công"

    def login(self, username, password):
        """
        Đăng nhập người dùng
        Args:
            username: Tên đăng nhập
            password: Mật khẩu
        Returns:
            tuple: (success, message) - Trạng thái đăng nhập và thông báo
        """
        # Kiểm tra thông tin đăng nhập
        if username not in self.users:
            return False, "Tên đăng nhập không tồn tại"
        if check_password_hash(self.users[username]['password'], password):
            session['username'] = username
            return True, "Đăng nhập thành công"
        return False, "Mật khẩu không đúng"

    def logout(self):
        """
        Đăng xuất người dùng
        """
        session.pop("username", None)
        return True, "Logout successful"

    def is_logged_in(self):
        """
        Kiểm tra trạng thái đăng nhập
        Returns:
            bool: True nếu đã đăng nhập, False nếu chưa
        """
        return "username" in session 