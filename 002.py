import os
import cv2
import sqlite3
import glob
import shutil
import pickle
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, abort
import base64
from PIL import Image
import io
import sys
import numpy as np

# Thêm import cho facenet_pytorch (chỉ import khi cần thiết)
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
    FACENET_AVAILABLE = True
except ImportError:
    print("Warning: facenet_pytorch not available. Face embedding updates will be disabled.")
    FACENET_AVAILABLE = False

# === CẤU HÌNH ===
DATA_ROOT = "data_files"
DB_PATH = os.path.join(DATA_ROOT, "face_recognition.db")
EMBEDDINGS_DIR = os.path.join(DATA_ROOT, "embeddings")
FACES_DIR = os.path.join(DATA_ROOT, "faces")

# Khởi tạo Flask app
app = Flask(__name__)

# Hàm để đảm bảo các thư mục và DB tồn tại
def ensure_initialized():
    # Tạo thư mục
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    
    # Tạo DB nếu chưa tồn tại
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tạo bảng users
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    )
    ''')
    
    # Tạo bảng faceIds
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faceIds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding_path TEXT NOT NULL,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Kiểm tra xem bảng face_images đã tồn tại chưa
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_images'")
    if not cursor.fetchone():
        # Tạo bảng face_images để lưu thông tin ảnh khuôn mặt
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            md5_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (face_id) REFERENCES faceIds(id)
        )
        ''')
        
        # Tạo index cho md5_hash để tìm kiếm nhanh
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_face_images_md5 ON face_images(md5_hash)
        ''')
    
    conn.commit()
    conn.close()

# Hàm lấy tất cả face từ DB
def get_all_faces():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Để lấy dữ liệu theo tên cột
    cursor = conn.cursor()
    
    # Lấy tất cả faces kèm thông tin user (nếu có)
    cursor.execute('''
    SELECT f.id as face_id, f.user_id, u.name as user_name
    FROM faceIds f
    LEFT JOIN users u ON f.user_id = u.id
    ORDER BY f.id
    ''')
    
    faces = []
    for row in cursor.fetchall():
        face_id = row['face_id']
        face_dir = os.path.join(FACES_DIR, f"face_{face_id}")
        
        # Kiểm tra xem có bảng face_images không
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_images'")
        if cursor.fetchone():
            # Lấy ảnh đầu tiên từ bảng face_images
            cursor.execute('''
            SELECT file_path FROM face_images 
            WHERE face_id = ? 
            ORDER BY created_at DESC 
            LIMIT 1
            ''', (face_id,))
            
            img_result = cursor.fetchone()
            face_img_path = img_result['file_path'] if img_result else None
        else:
            # Lấy ảnh đầu tiên theo cách cũ
            face_images = glob.glob(os.path.join(face_dir, "*.jpg"))
            if not face_images:
                face_images = glob.glob(os.path.join(face_dir, "*.png"))
                
            face_img_path = face_images[0] if face_images else None
        
        faces.append({
            'face_id': face_id,
            'user_id': row['user_id'],
            'user_name': row['user_name'],
            'image_path': face_img_path
        })
    
    conn.close()
    return faces

# Hàm lấy thông tin của một face cụ thể
def get_face(face_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT f.id as face_id, f.user_id, u.name as user_name
    FROM faceIds f
    LEFT JOIN users u ON f.user_id = u.id
    WHERE f.id = ?
    ''', (face_id,))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    
    # Kiểm tra xem có bảng face_images không
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_images'")
    if cursor.fetchone():
        # Lấy ảnh đầu tiên từ bảng face_images
        cursor.execute('''
        SELECT file_path FROM face_images 
        WHERE face_id = ? 
        ORDER BY created_at DESC 
        LIMIT 1
        ''', (face_id,))
        
        img_result = cursor.fetchone()
        face_img_path = img_result['file_path'] if img_result else None
    else:
        # Lấy ảnh đầu tiên theo cách cũ
        face_dir = os.path.join(FACES_DIR, f"face_{face_id}")
        face_images = glob.glob(os.path.join(face_dir, "*.jpg"))
        if not face_images:
            face_images = glob.glob(os.path.join(face_dir, "*.png"))
            
        face_img_path = face_images[0] if face_images else None
    
    face = {
        'face_id': row['face_id'],
        'user_id': row['user_id'],
        'user_name': row['user_name'],
        'image_path': face_img_path
    }
    
    conn.close()
    return face

# Hàm lấy tất cả ảnh của một face
def get_face_images(face_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Kiểm tra xem có bảng face_images không
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_images'")
    
    if cursor.fetchone():
        # Lấy từ bảng face_images
        cursor.execute('''
        SELECT id, file_path, md5_hash, created_at
        FROM face_images
        WHERE face_id = ?
        ORDER BY created_at DESC
        ''', (face_id,))
        
        images = []
        for row in cursor.fetchall():
            images.append({
                'id': row['id'],
                'file_path': row['file_path'],
                'md5_hash': row['md5_hash'],
                'created_at': row['created_at']
            })
    else:
        # Lấy ảnh theo cách cũ
        face_dir = os.path.join(FACES_DIR, f"face_{face_id}")
        face_images = glob.glob(os.path.join(face_dir, "*.jpg"))
        face_images.extend(glob.glob(os.path.join(face_dir, "*.png")))
        
        images = []
        for img_path in face_images:
            images.append({
                'id': None,  # Không có ID trong file system
                'file_path': img_path,
                'md5_hash': None,  # Không có MD5 hash
                'created_at': None  # Không có created_at
            })
    
    conn.close()
    return images

# Hàm lấy danh sách tất cả users
def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name FROM users ORDER BY name')
    
    users = []
    for row in cursor.fetchall():
        users.append({
            'id': row['id'],
            'name': row['name']
        })
    
    conn.close()
    return users

# Hàm xóa một ảnh
def delete_face_image(image_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Kiểm tra xem có bảng face_images không
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_images'")
    if not cursor.fetchone():
        conn.close()
        return False, "Face images table does not exist"
    
    # Lấy thông tin ảnh trước khi xóa
    cursor.execute("SELECT file_path FROM face_images WHERE id = ?", (image_id,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return False, "Image not found"
    
    file_path = result[0]
    
    try:
        # Xóa file ảnh nếu tồn tại
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Xóa bản ghi trong DB
        cursor.execute("DELETE FROM face_images WHERE id = ?", (image_id,))
        conn.commit()
        
        conn.close()
        return True, f"Deleted image {image_id}"
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, str(e)

# Hàm gán ảnh cho một face_id khác
def reassign_face_image(image_id, new_face_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Kiểm tra xem new_face_id có tồn tại không
        cursor.execute("SELECT id FROM faceIds WHERE id = ?", (new_face_id,))
        if not cursor.fetchone():
            conn.close()
            return False, f"Face ID {new_face_id} does not exist"
        
        # Lấy thông tin ảnh
        cursor.execute("SELECT file_path, md5_hash, face_id FROM face_images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False, "Image not found"
        
        old_file_path, md5_hash, old_face_id = result
        
        # Tạo đường dẫn mới cho file
        old_filename = os.path.basename(old_file_path)
        new_face_dir = os.path.join(FACES_DIR, f"face_{new_face_id}")
        os.makedirs(new_face_dir, exist_ok=True)
        new_file_path = os.path.join(new_face_dir, old_filename)
        
        # Di chuyển file
        if os.path.exists(old_file_path):
            print(f"Moving file from {old_file_path} to {new_file_path}")
            # Copy trước rồi xóa sau để tránh lỗi nếu thao tác di chuyển bị gián đoạn
            shutil.copy2(old_file_path, new_file_path)
            os.remove(old_file_path)
        else:
            print(f"Warning: Source file {old_file_path} does not exist")
        
        # Cập nhật bản ghi trong DB
        cursor.execute(
            "UPDATE face_images SET face_id = ?, file_path = ? WHERE id = ?", 
            (new_face_id, new_file_path, image_id)
        )
        
        conn.commit()
        conn.close()
        return True, f"Reassigned image {image_id} to face ID {new_face_id}"
    except Exception as e:
        import traceback
        print(f"Error in reassign_face_image: {e}")
        print(traceback.format_exc())
        conn.rollback()
        conn.close()
        return False, str(e)

# Hàm gán tên cho face
def assign_name_to_face(face_id, user_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Kiểm tra user_name đã tồn tại chưa
    cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
    result = cursor.fetchone()
    
    user_id = None
    if result:
        user_id = result[0]
    else:
        # Tạo user mới
        cursor.execute("INSERT INTO users (name) VALUES (?)", (user_name,))
        user_id = cursor.lastrowid
    
    # Cập nhật face
    cursor.execute("UPDATE faceIds SET user_id = ? WHERE id = ?", (user_id, face_id))
    
    conn.commit()
    conn.close()
    
    return user_id

# Đọc ảnh và chuyển thành base64 để hiển thị trên web
def get_image_base64(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        encoded = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

# Route trang chủ: hiển thị tất cả faces
@app.route('/')
def home():
    ensure_initialized()
    faces = get_all_faces()
    
    # Thêm dữ liệu base64 cho mỗi ảnh
    for face in faces:
        if face['image_path']:
            face['image_data'] = get_image_base64(face['image_path'])
        else:
            face['image_data'] = None
    
    return render_template('index.html', faces=faces)

# Route hiển thị chi tiết face
@app.route('/face/<int:face_id>')
def face_detail(face_id):
    face = get_face(face_id)
    if not face:
        abort(404)
    
    # Lấy tất cả ảnh của face
    images = get_face_images(face_id)
    
    # Thêm dữ liệu base64 cho mỗi ảnh
    for img in images:
        if img['file_path'] and os.path.exists(img['file_path']):
            img['image_data'] = get_image_base64(img['file_path'])
        else:
            img['image_data'] = None
    
    # Lấy danh sách tất cả users để gán lại
    users = get_all_users()
    
    return render_template('face_detail.html', face=face, images=images, users=users)

# API gán tên cho face
@app.route('/assign_name', methods=['POST'])
def assign_name():
    face_id = request.form.get('face_id')
    user_name = request.form.get('user_name')
    
    if not face_id or not user_name:
        return jsonify({'status': 'error', 'message': 'Missing face_id or user_name'}), 400
    
    try:
        user_id = assign_name_to_face(int(face_id), user_name)
        return jsonify({'status': 'success', 'user_id': user_id})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Route lấy ảnh face
@app.route('/face_image/<int:face_id>')
def face_image(face_id):
    face = get_face(face_id)
    if not face or not face['image_path'] or not os.path.exists(face['image_path']):
        return "Image not found", 404
    
    return send_file(face['image_path'])

# Route cập nhật tên theo batch
@app.route('/batch_update', methods=['POST'])
def batch_update():
    data = request.json
    if not data or 'updates' not in data:
        return jsonify({'status': 'error', 'message': 'No update data provided'}), 400
    
    updates = data['updates']
    results = []
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        for update in updates:
            face_id = update.get('face_id')
            user_name = update.get('user_name')
            
            if not face_id or not user_name:
                continue
            
            # Tìm hoặc tạo người dùng
            cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
            result = cursor.fetchone()
            
            user_id = None
            if result:
                user_id = result[0]
            else:
                cursor.execute("INSERT INTO users (name) VALUES (?)", (user_name,))
                user_id = cursor.lastrowid
            
            # Cập nhật face
            cursor.execute("UPDATE faceIds SET user_id = ? WHERE id = ?", (user_id, face_id))
            
            results.append({
                'face_id': face_id,
                'user_id': user_id,
                'user_name': user_name,
                'status': 'success'
            })
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()
    
    return jsonify({'status': 'success', 'results': results})

# Route xóa faces
@app.route('/delete_faces', methods=['POST'])
def delete_faces_route():
    data = request.json
    if not data or 'face_ids' not in data:
        return jsonify({'status': 'error', 'message': 'No face_ids provided'}), 400
    
    face_ids = data['face_ids']
    if not face_ids:
        return jsonify({'status': 'error', 'message': 'Empty face_ids list'}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        results = []
        for face_id in face_ids:
            try:
                # Lấy embedding_path
                cursor.execute("SELECT embedding_path FROM faceIds WHERE id = ?", (face_id,))
                result = cursor.fetchone()
                if not result:
                    results.append({'face_id': face_id, 'status': 'error', 'message': 'Face not found'})
                    continue
                    
                embedding_path = result[0]
                
                # Lấy tất cả ảnh từ bảng face_images
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_images'")
                if cursor.fetchone():
                    cursor.execute("SELECT id, file_path FROM face_images WHERE face_id = ?", (face_id,))
                    images = cursor.fetchall()
                    
                    # Xóa tất cả ảnh
                    for image_id, file_path in images:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        cursor.execute("DELETE FROM face_images WHERE id = ?", (image_id,))
                
                # Xóa thư mục chứa ảnh
                face_dir = os.path.join(FACES_DIR, f"face_{face_id}")
                if os.path.exists(face_dir):
                    shutil.rmtree(face_dir)
                
                # Xóa file embedding
                if embedding_path and os.path.exists(embedding_path):
                    os.remove(embedding_path)
                
                # Xóa bản ghi trong DB
                cursor.execute("DELETE FROM faceIds WHERE id = ?", (face_id,))
                
                results.append({'face_id': face_id, 'status': 'success'})
            except Exception as e:
                results.append({'face_id': face_id, 'status': 'error', 'message': str(e)})
        
        # Xóa users không còn face nào
        cursor.execute('''
        DELETE FROM users 
        WHERE id NOT IN (SELECT DISTINCT user_id FROM faceIds WHERE user_id IS NOT NULL)
        ''')
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Route xóa ảnh
@app.route('/delete_image/<int:image_id>', methods=['POST'])
def delete_image_route(image_id):
    success, message = delete_face_image(image_id)
    return jsonify({'status': 'success' if success else 'error', 'message': message})

# Route gán lại ảnh
@app.route('/reassign_image', methods=['POST'])
def reassign_image_route():
    image_id = request.form.get('image_id')
    user_id = request.form.get('user_id')  # Nhận user_id thay vì face_id
    new_person_name = request.form.get('new_person_name')
    
    print(f"Reassign request - image_id: {image_id}, user_id: {user_id}, new_person_name: {new_person_name}")
    
    if not image_id:
        return jsonify({'status': 'error', 'message': 'Missing image_id'}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Lấy thông tin ảnh
        cursor.execute("SELECT file_path, md5_hash, face_id FROM face_images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'status': 'error', 'message': 'Image not found'}), 404
            
        old_file_path, md5_hash, source_face_id = result['file_path'], result['md5_hash'], result['face_id']
        
        # Lấy thông tin source face
        cursor.execute("SELECT user_id FROM faceIds WHERE id = ?", (source_face_id,))
        source_face = cursor.fetchone()
        source_user_id = source_face['user_id'] if source_face else None
        
        # Trường hợp 1: Tạo người mới
        if new_person_name:
            print(f"Creating new person: {new_person_name}")
            
            # Tạo user mới
            cursor.execute("INSERT INTO users (name) VALUES (?)", (new_person_name,))
            target_user_id = cursor.lastrowid
            
            # Tạo face mới cho user mới
            embedding_filename = f"embedding_{uuid.uuid4()}.pkl"
            embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_filename)
            
            # Tạo placeholder embedding
            with open(embedding_path, 'wb') as f:
                placeholder_embedding = [0.0] * 512
                pickle.dump(placeholder_embedding, f)
            
            # Tạo bản ghi face mới
            cursor.execute(
                "INSERT INTO faceIds (embedding_path, user_id) VALUES (?, ?)",
                (embedding_path, target_user_id)
            )
            target_face_id = cursor.lastrowid
            
            print(f"Created new face_id {target_face_id} for user_id {target_user_id}")
            
            # Tạo thư mục mới
            target_dir = os.path.join(FACES_DIR, f"face_{target_face_id}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Di chuyển ảnh sang thư mục mới
            file_name = os.path.basename(old_file_path)
            new_file_path = os.path.join(target_dir, file_name)
            
            if os.path.exists(old_file_path):
                print(f"Moving file from {old_file_path} to {new_file_path}")
                shutil.copy2(old_file_path, new_file_path)
                os.remove(old_file_path)
            
            # Cập nhật bản ghi ảnh
            cursor.execute(
                "UPDATE face_images SET face_id = ?, file_path = ? WHERE id = ?", 
                (target_face_id, new_file_path, image_id)
            )
            
            message = f"Reassigned image to new user {new_person_name} (User ID: {target_user_id})"
        
        # Trường hợp 2: Gán cho user có sẵn
        elif user_id:
            target_user_id = int(user_id)
            print(f"Reassigning image {image_id} from user_id {source_user_id} to user_id {target_user_id}")
            
            # Kiểm tra user có tồn tại không
            cursor.execute("SELECT id, name FROM users WHERE id = ?", (target_user_id,))
            user_result = cursor.fetchone()
            if not user_result:
                conn.close()
                return jsonify({'status': 'error', 'message': f'User ID {target_user_id} does not exist'}), 400
                
            user_name = user_result['name']
            
            # Kiểm tra xem user có face_id nào không
            cursor.execute("SELECT id FROM faceIds WHERE user_id = ? LIMIT 1", (target_user_id,))
            face_result = cursor.fetchone()
            
            if face_result:
                # Sử dụng face_id có sẵn của user
                target_face_id = face_result['id']
                print(f"Using existing face_id {target_face_id} for user_id {target_user_id}")
            else:
                # Tạo face_id mới cho user
                embedding_filename = f"embedding_{uuid.uuid4()}.pkl"
                embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_filename)
                
                # Tạo placeholder embedding
                with open(embedding_path, 'wb') as f:
                    placeholder_embedding = [0.0] * 512
                    pickle.dump(placeholder_embedding, f)
                
                # Tạo bản ghi face mới
                cursor.execute(
                    "INSERT INTO faceIds (embedding_path, user_id) VALUES (?, ?)",
                    (embedding_path, target_user_id)
                )
                target_face_id = cursor.lastrowid
                print(f"Created new face_id {target_face_id} for user_id {target_user_id}")
            
            # Tạo thư mục cho face_id nếu chưa có
            target_dir = os.path.join(FACES_DIR, f"face_{target_face_id}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Di chuyển ảnh sang thư mục mới
            file_name = os.path.basename(old_file_path)
            new_file_path = os.path.join(target_dir, file_name)
            
            if os.path.exists(old_file_path):
                print(f"Moving file from {old_file_path} to {new_file_path}")
                shutil.copy2(old_file_path, new_file_path)
                os.remove(old_file_path)
            else:
                print(f"Warning: Source file {old_file_path} does not exist")
            
            # Cập nhật bản ghi ảnh
            cursor.execute(
                "UPDATE face_images SET face_id = ?, file_path = ? WHERE id = ?", 
                (target_face_id, new_file_path, image_id)
            )
            
            message = f"Reassigned image to user {user_name} (User ID: {target_user_id})"
        
        else:
            conn.close()
            return jsonify({'status': 'error', 'message': 'Missing user_id or new_person_name'}), 400
        
        # Sau khi đã cập nhật DB và di chuyển ảnh, thêm phần cập nhật embedding:
        
        # 1. Cập nhật embedding cho face_id đích
        try:
            # Lấy tất cả ảnh của face_id đích
            cursor.execute("SELECT file_path FROM face_images WHERE face_id = ?", (target_face_id,))
            target_images = cursor.fetchall()
            
            if target_images and len(target_images) > 0:
                # Lấy đường dẫn embedding
                cursor.execute("SELECT embedding_path FROM faceIds WHERE id = ?", (target_face_id,))
                embedding_result = cursor.fetchone()
                if embedding_result:
                    embedding_path = embedding_result['embedding_path']
                    
                    # Tạo danh sách đường dẫn ảnh
                    image_paths = [img['file_path'] for img in target_images]
                    
                    # Gọi hàm cập nhật embedding (giả sử bạn có hàm này)
                    if FACENET_AVAILABLE:
                        print(f"Updating embedding for target face_id {target_face_id} with {len(image_paths)} images")
                        update_face_embedding(target_face_id, image_paths, embedding_path)
        except Exception as e:
            print(f"Warning: Failed to update target embedding: {e}")
            # Không throw exception ở đây để không ảnh hưởng đến flow chính
        
        # 2. Cập nhật embedding cho face_id nguồn (nếu còn ảnh)
        try:
            # Kiểm tra xem face_id nguồn còn ảnh nào không
            cursor.execute("SELECT COUNT(*) as count FROM face_images WHERE face_id = ?", (source_face_id,))
            count_result = cursor.fetchone()
            
            if count_result and count_result['count'] > 0:
                # Lấy tất cả ảnh còn lại của face_id nguồn
                cursor.execute("SELECT file_path FROM face_images WHERE face_id = ?", (source_face_id,))
                source_images = cursor.fetchall()
                
                # Lấy đường dẫn embedding
                cursor.execute("SELECT embedding_path FROM faceIds WHERE id = ?", (source_face_id,))
                embedding_result = cursor.fetchone()
                
                if embedding_result:
                    embedding_path = embedding_result['embedding_path']
                    
                    # Tạo danh sách đường dẫn ảnh
                    image_paths = [img['file_path'] for img in source_images]
                    
                    # Gọi hàm cập nhật embedding
                    if FACENET_AVAILABLE:
                        print(f"Updating embedding for source face_id {source_face_id} with {len(image_paths)} images")
                        update_face_embedding(source_face_id, image_paths, embedding_path)
            else:
                # Nếu face_id nguồn không còn ảnh nào, có thể xem xét xóa
                print(f"Source face_id {source_face_id} has no more images. Consider removing it.")
                # Tùy vào logic ứng dụng, bạn có thể quyết định xóa face_id này
        except Exception as e:
            print(f"Warning: Failed to update source embedding: {e}")
            # Không throw exception ở đây
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': message})
        
    except Exception as e:
        import traceback
        print(f"Error in reassign_image_route: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500

def update_face_embedding(face_id, image_paths, embedding_path):
    """
    Cập nhật embedding cho face_id dựa trên các ảnh sử dụng facenet_pytorch
    
    Args:
        face_id: ID của khuôn mặt cần cập nhật
        image_paths: Danh sách đường dẫn đến các ảnh của face này
        embedding_path: Đường dẫn đến file embedding
    """
    if not image_paths:
        print(f"No images for face_id {face_id}, skipping embedding update")
        return
        
    try:
        # Import các thư viện cần thiết
        from facenet_pytorch import MTCNN, InceptionResnetV1
        import torch
        from PIL import Image
        import numpy as np
        
        # Khởi tạo các model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )
        
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Danh sách lưu các embedding của từng ảnh
        all_embeddings = []
        
        # Trích xuất embedding từ mỗi ảnh
        for img_path in image_paths:
            if os.path.exists(img_path):
                try:
                    # Đọc ảnh bằng PIL
                    img = Image.open(img_path)
                    
                    # Phát hiện và căn chỉnh khuôn mặt
                    face = mtcnn(img)
                    
                    if face is not None:
                        # Tính toán embedding
                        with torch.no_grad():
                            face_embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]
                        all_embeddings.append(face_embedding)
                    else:
                        print(f"No face detected in {img_path}")
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            else:
                print(f"Image {img_path} does not exist")
        
        if all_embeddings:
            # Tính trung bình các embedding
            average_embedding = np.mean(all_embeddings, axis=0)
            
            # Lưu embedding vào file
            with open(embedding_path, 'wb') as f:
                pickle.dump(average_embedding, f)
            
            print(f"Updated embedding for face_id {face_id} with {len(all_embeddings)} images")
        else:
            print(f"No valid embeddings generated for face_id {face_id}")
            
            # Nếu không có embedding hợp lệ, tạo embedding rỗng
            with open(embedding_path, 'wb') as f:
                # Embedding size cho InceptionResnetV1 là 512
                placeholder_embedding = np.zeros(512, dtype=np.float32)
                pickle.dump(placeholder_embedding, f)
    
    except Exception as e:
        import traceback
        print(f"Error updating face embedding: {e}")
        print(traceback.format_exc())

# Template cho trang index.html (sẽ tạo nếu chưa có)
def create_template_files():
    os.makedirs('templates', exist_ok=True)
    
    # Tạo file index.html
    index_html = '''
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quản lý khuôn mặt</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            padding-top: 80px; /* Để chừa chỗ cho floating menu */
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .face-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
        }
        .face-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s;
            position: relative;
        }
        .face-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .face-image-container {
            position: relative;
            width: 100%;
            height: 200px;
            cursor: pointer;
        }
        .face-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .face-info {
            padding: 10px;
        }
        .face-id {
            font-weight: bold;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .detail-btn {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 3px 8px;
            cursor: pointer;
            font-size: 12px;
            text-decoration: none;
        }
        .detail-btn:hover {
            background-color: #5a6268;
        }
        .face-name {
            margin-bottom: 10px;
            min-height: 20px;
        }
        .name-input {
            width: 100%;
            padding: 8px;
            margin-bottom: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .assign-btn {
            width: 100%;
            padding: 8px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .assign-btn:hover {
            background-color: #45a049;
        }
        .floating-menu {
            position: fixed;
            top: 0;
            right: 0;
            left: 0;
            background-color: #fff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            z-index: 1000;
            display: flex;
            justify-content: flex-end;
            align-items: center;
        }
        .floating-menu button {
            padding: 10px 20px;
            margin-left: 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .save-all-btn {
            background-color: #007bff;
            color: white;
        }
        .save-all-btn:hover {
            background-color: #0069d9;
        }
        .delete-btn {
            background-color: #dc3545;
            color: white;
        }
        .delete-btn:hover {
            background-color: #c82333;
        }
        .status-message {
            margin-right: 20px;
            font-weight: bold;
        }
        .status-success {
            color: #28a745;
        }
        .status-error {
            color: #dc3545;
        }
        .checkbox-container {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
        }
        .face-checkbox {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        .select-all-container {
            margin-right: auto;
            display: flex;
            align-items: center;
        }
        .select-all-container input {
            margin-right: 5px;
            width: 20px;
            height: 20px;
        }
        #selectedCount {
            margin-left: 15px;
            font-weight: bold;
        }
        /* Thêm hiệu ứng khi ảnh được chọn */
        .face-image-container.selected::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 123, 255, 0.3);
            pointer-events: none;
        }
    </style>
</head>
<body>
    <h1>Quản lý khuôn mặt</h1>
    
    <div class="floating-menu">
        <div class="select-all-container">
            <input type="checkbox" id="selectAll">
            <label for="selectAll">Chọn tất cả</label>
            <span id="selectedCount"></span>
        </div>
        
        <div id="statusMessage" class="status-message"></div>
        <button id="saveAllBtn" class="save-all-btn">Lưu tất cả thay đổi</button>
        <button id="deleteBtn" class="delete-btn">Xóa đã chọn</button>
    </div>
    
    <div class="face-grid">
        {% for face in faces %}
            <div class="face-card" data-face-id="{{ face.face_id }}">
                <div class="face-image-container">
                    <div class="checkbox-container">
                        <input type="checkbox" class="face-checkbox" data-face-id="{{ face.face_id }}">
                    </div>
                    
                    {% if face.image_data %}
                        <img src="{{ face.image_data }}" alt="Face {{ face.face_id }}" class="face-image">
                    {% else %}
                        <div class="face-image" style="background-color: #eee; display: flex; align-items: center; justify-content: center;">
                            <span>No image</span>
                        </div>
                    {% endif %}
                </div>
                <div class="face-info">
                    <div class="face-id">
                        <span>ID: {{ face.face_id }}</span>
                        <a href="/face/{{ face.face_id }}" class="detail-btn">Chi tiết</a>
                    </div>
                    <div class="face-name">{{ face.user_name or 'Unknown' }}</div>
                    <input type="text" class="name-input" value="{{ face.user_name or '' }}" placeholder="Nhập tên...">
                    <button class="assign-btn" onclick="assignName({{ face.face_id }})">Gán tên</button>
                </div>
            </div>
        {% endfor %}
    </div>
    
    <script>
        // Lưu những thay đổi chưa lưu
        const pendingChanges = {};
        
        // Hàm gán tên cho một khuôn mặt
        function assignName(faceId) {
            const card = document.querySelector(`.face-card[data-face-id="${faceId}"]`);
            const nameInput = card.querySelector('.name-input');
            const userName = nameInput.value.trim();
            
            if (!userName) {
                alert('Vui lòng nhập tên!');
                return;
            }
            
            fetch('/assign_name', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `face_id=${faceId}&user_name=${encodeURIComponent(userName)}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    card.querySelector('.face-name').textContent = userName;
                    showStatus(`Đã gán ID ${faceId} cho "${userName}"`, 'success');
                    
                    // Xóa khỏi pending changes nếu đã lưu
                    if (pendingChanges[faceId]) {
                        delete pendingChanges[faceId];
                    }
                } else {
                    showStatus(`Lỗi: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Lỗi: ${error.message}`, 'error');
            });
        }
        
        // Theo dõi các thay đổi trong name input
        document.querySelectorAll('.name-input').forEach(input => {
            const faceId = input.closest('.face-card').dataset.faceId;
            const originalName = input.closest('.face-info').querySelector('.face-name').textContent;
            
            input.addEventListener('input', () => {
                const newName = input.value.trim();
                // Chỉ lưu vào pending nếu khác tên ban đầu
                if (newName && newName !== originalName && newName !== 'Unknown') {
                    pendingChanges[faceId] = newName;
                } else {
                    delete pendingChanges[faceId];
                }
            });
        });
        
        // Lưu tất cả thay đổi
        document.getElementById('saveAllBtn').addEventListener('click', () => {
            const updates = Object.entries(pendingChanges).map(([faceId, userName]) => ({
                face_id: parseInt(faceId),
                user_name: userName
            }));
            
            if (updates.length === 0) {
                showStatus('Không có thay đổi để lưu', 'error');
                return;
            }
            
            showStatus('Đang lưu...', '');
            
            fetch('/batch_update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ updates })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`Đã lưu ${updates.length} thay đổi thành công!`, 'success');
                    
                    // Cập nhật UI
                    data.results.forEach(result => {
                        const card = document.querySelector(`.face-card[data-face-id="${result.face_id}"]`);
                        if (card) {
                            card.querySelector('.face-name').textContent = result.user_name;
                        }
                    });
                    
                    // Xóa pendingChanges
                    Object.keys(pendingChanges).forEach(key => delete pendingChanges[key]);
                } else {
                    showStatus(`Lỗi: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Lỗi: ${error.message}`, 'error');
            });
        });
        
        // Chọn tất cả checkboxes
        document.getElementById('selectAll').addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('.face-checkbox');
            checkboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
                updateImageContainerClass(checkbox);
            });
            updateSelectedCount();
        });
        
        // Cập nhật số lượng đã chọn
        function updateSelectedCount() {
            const selectedCount = document.querySelectorAll('.face-checkbox:checked').length;
            document.getElementById('selectedCount').textContent = selectedCount > 0 ? 
                `(Đã chọn ${selectedCount})` : '';
        }
        
        // Cập nhật class cho container khi checkbox thay đổi
        function updateImageContainerClass(checkbox) {
            const container = checkbox.closest('.face-card').querySelector('.face-image-container');
            if (checkbox.checked) {
                container.classList.add('selected');
            } else {
                container.classList.remove('selected');
            }
        }
        
        // Theo dõi trạng thái checkbox từng face
        document.querySelectorAll('.face-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                updateImageContainerClass(checkbox);
                updateSelectedCount();
                // Kiểm tra nếu tất cả đều được chọn thì select all cũng phải được check
                const allChecked = document.querySelectorAll('.face-checkbox:not(:checked)').length === 0;
                document.getElementById('selectAll').checked = allChecked;
            });
        });
        
        // Xử lý click vào ảnh để toggle checkbox
        document.querySelectorAll('.face-image-container').forEach(container => {
            container.addEventListener('click', function(e) {
                // Ngăn xung đột khi click vào checkbox
                if (e.target.classList.contains('face-checkbox')) {
                    return;
                }
                
                const faceId = this.closest('.face-card').dataset.faceId;
                const checkbox = document.querySelector(`.face-checkbox[data-face-id="${faceId}"]`);
                
                // Toggle checkbox
                checkbox.checked = !checkbox.checked;
                
                // Kích hoạt sự kiện change để cập nhật UI
                const event = new Event('change');
                checkbox.dispatchEvent(event);
            });
        });
        
        // Xóa faces đã chọn
        document.getElementById('deleteBtn').addEventListener('click', () => {
            const selectedCheckboxes = document.querySelectorAll('.face-checkbox:checked');
            if (selectedCheckboxes.length === 0) {
                showStatus('Vui lòng chọn ít nhất một khuôn mặt để xóa', 'error');
                return;
            }
            
            if (!confirm(`Bạn có chắc chắn muốn xóa ${selectedCheckboxes.length} khuôn mặt đã chọn không?`)) {
                return;
            }
            
            const faceIds = Array.from(selectedCheckboxes).map(
                checkbox => parseInt(checkbox.dataset.faceId)
            );
            
            showStatus('Đang xóa...', '');
            
            fetch('/delete_faces', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ face_ids: faceIds })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`Đã xóa ${data.results.filter(r => r.status === 'success').length} khuôn mặt thành công!`, 'success');
                    
                    // Xóa card khỏi UI
                    data.results.forEach(result => {
                        if (result.status === 'success') {
                            const card = document.querySelector(`.face-card[data-face-id="${result.face_id}"]`);
                            if (card) {
                                card.remove();
                            }
                        }
                    });
                    
                    // Cập nhật count
                    updateSelectedCount();
                    
                    // Nếu không còn face nào, bỏ chọn select all
                    if (document.querySelectorAll('.face-card').length === 0) {
                        document.getElementById('selectAll').checked = false;
                    }
                    
                    // Tải lại trang sau 1 giây
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                } else {
                    showStatus(`Lỗi: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Lỗi: ${error.message}`, 'error');
            });
        });
        
        function showStatus(message, type) {
            const statusElement = document.getElementById('statusMessage');
            statusElement.textContent = message;
            statusElement.className = 'status-message';
            
            if (type) {
                statusElement.classList.add(`status-${type}`);
            }
            
            // Tự động ẩn sau 5 giây nếu là thông báo thành công
            if (type === 'success') {
                setTimeout(() => {
                    statusElement.textContent = '';
                    statusElement.className = 'status-message';
                }, 5000);
            }
        }
        
        // Khởi tạo số lượng đã chọn và class cho các container
        updateSelectedCount();
        document.querySelectorAll('.face-checkbox').forEach(updateImageContainerClass);
    </script>
</body>
</html>
    '''
    
    # Tạo file face_detail.html
    face_detail_html = '''
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chi tiết khuôn mặt - ID: {{ face.face_id }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .face-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        .face-header-image {
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: 50%;
            margin-right: 20px;
        }
        .face-info {
            flex: 1;
        }
        .face-info h2 {
            margin: 0 0 10px 0;
        }
        .face-info p {
            margin: 5px 0;
            color: #666;
        }
        .back-btn {
            display: inline-block;
            padding: 8px 16px;
            background-color: #6c757d;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .back-btn:hover {
            background-color: #5a6268;
        }
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        .image-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .face-image {
            width: 100%;
            height: 250px;
            object-fit: cover;
        }
        .image-actions {
            padding: 10px;
        }
        .delete-img-btn {
            width: 100%;
            padding: 8px;
            background-color: #dc3545;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 10px;
        }
        .delete-img-btn:hover {
            background-color: #c82333;
        }
        .reassign-container {
            display: flex;
            gap: 5px;
        }
        .reassign-select {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .reassign-btn {
            padding: 8px;
            background-color: #ffc107;
            color: black;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .reassign-btn:hover {
            background-color: #e0a800;
        }
        .image-time {
            font-size: 12px;
            color: #6c757d;
            margin: 5px 0;
        }
        .status-message {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 4px;
            font-weight: bold;
            z-index: 1000;
            display: none;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .empty-message {
            grid-column: 1 / -1;
            text-align: center;
            padding: 20px;
            color: #6c757d;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div id="statusMessage" class="status-message"></div>
    
    <a href="/" class="back-btn">← Quay lại</a>
    
    <div class="face-header">
        {% if face.image_path %}
            <img src="{{ face.image_data }}" alt="Face {{ face.face_id }}" class="face-header-image">
        {% endif %}
        <div class="face-info">
            <h2>{% if face.user_name %}{{ face.user_name }}{% else %}Chưa có tên{% endif %}</h2>
            <p>ID: {{ face.face_id }}</p>
            {% if face.user_id %}
                <p>User ID: {{ face.user_id }}</p>
            {% endif %}
        </div>
    </div>
    
    <h1>Tất cả ảnh của khuôn mặt</h1>
    
    <div class="images-grid">
        {% if images %}
            {% for img in images %}
                <div class="image-card" data-image-id="{{ img.id }}">
                    {% if img.image_data %}
                        <img src="{{ img.image_data }}" alt="Face image" class="face-image">
                    {% else %}
                        <div class="face-image" style="background-color: #eee; display: flex; align-items: center; justify-content: center;">
                            <span>No image</span>
                        </div>
                    {% endif %}
                    <div class="image-actions">
                        {% if img.created_at %}
                            <div class="image-time">Thời gian: {{ img.created_at }}</div>
                        {% endif %}
                        
                        {% if img.id is not none %}
                            <button class="delete-img-btn" onclick="deleteImage({{ img.id }})">Xóa ảnh</button>
                            
                            <div class="reassign-container">
                                <select class="reassign-select" onchange="toggleNewPersonInput(this)">
                                    <option value="">-- Chọn người khác --</option>
                                    <option value="new">++ Người mới ++</option>
                                    {% for user in users %}
                                        <option value="{{ user.id }}" {% if user.id == face.user_id %}disabled{% endif %}>
                                            {{ user.name }}
                                        </option>
                                    {% endfor %}
                                </select>
                                <input type="text" class="new-person-input" placeholder="Nhập tên người mới" style="display: none; flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                                <button class="reassign-btn" onclick="reassignImage({{ img.id }}, this)">Gán</button>
                            </div>
                        {% endif %}
                    </div>
                </div>
            {% endfor %}
        {% else %}
            <div class="empty-message">Không có ảnh nào cho khuôn mặt này</div>
        {% endif %}
    </div>
    
    <script>
        function showStatus(message, type) {
            const statusElement = document.getElementById('statusMessage');
            statusElement.textContent = message;
            statusElement.className = 'status-message';
            
            if (type) {
                statusElement.classList.add(`status-${type}`);
            }
            
            statusElement.style.display = 'block';
            
            // Tự động ẩn sau 3 giây
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 3000);
        }
        
        function deleteImage(imageId) {
            if (!confirm('Bạn có chắc chắn muốn xóa ảnh này không?')) {
                return;
            }
            
            fetch(`/delete_image/${imageId}`, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Xóa card khỏi UI
                    const card = document.querySelector(`.image-card[data-image-id="${imageId}"]`);
                    if (card) {
                        card.remove();
                    }
                    
                    // Kiểm tra nếu không còn ảnh nào
                    const remainingCards = document.querySelectorAll('.image-card');
                    if (remainingCards.length === 0) {
                        const grid = document.querySelector('.images-grid');
                        grid.innerHTML = '<div class="empty-message">Không có ảnh nào cho khuôn mặt này</div>';
                    }
                    
                    showStatus('Đã xóa ảnh thành công', 'success');
                } else {
                    showStatus(`Lỗi: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Lỗi: ${error.message}`, 'error');
            });
        }
        
        function toggleNewPersonInput(selectElement) {
            const container = selectElement.closest('.reassign-container');
            const newPersonInput = container.querySelector('.new-person-input');
            
            if (selectElement.value === 'new') {
                newPersonInput.style.display = 'block';
                selectElement.style.width = '40%';
            } else {
                newPersonInput.style.display = 'none';
                selectElement.style.width = '';
            }
        }

        function reassignImage(imageId, buttonElement) {
            const container = buttonElement.closest('.reassign-container');
            const select = container.querySelector('.reassign-select');
            const newPersonInput = container.querySelector('.new-person-input');
            
            let userId = select.value;  // Giờ đây chúng ta gọi nó là userId thay vì faceId
            let newPersonName = '';
            
            if (userId === 'new') {
                newPersonName = newPersonInput.value.trim();
                if (!newPersonName) {
                    showStatus('Vui lòng nhập tên người mới', 'error');
                    return;
                }
            } else if (!userId) {
                showStatus('Vui lòng chọn một người', 'error');
                return;
            }
            
            // Chuẩn bị dữ liệu gửi đi
            let formData;
            if (userId === 'new') {
                formData = `image_id=${imageId}&new_person_name=${encodeURIComponent(newPersonName)}`;
            } else {
                formData = `image_id=${imageId}&user_id=${userId}`;  // Thay đổi face_id thành user_id
            }
            
            fetch('/reassign_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Xóa card khỏi UI
                    const card = document.querySelector(`.image-card[data-image-id="${imageId}"]`);
                    if (card) {
                        card.remove();
                    }
                    
                    // Kiểm tra nếu không còn ảnh nào
                    const remainingCards = document.querySelectorAll('.image-card');
                    if (remainingCards.length === 0) {
                        const grid = document.querySelector('.images-grid');
                        grid.innerHTML = '<div class="empty-message">Không có ảnh nào cho khuôn mặt này</div>';
                    }
                    
                    showStatus('Đã gán ảnh thành công', 'success');
                } else {
                    showStatus(`Lỗi: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Lỗi: ${error.message}`, 'error');
            });
        }
    </script>
</body>
</html>
    '''
    
    with open(os.path.join('templates', 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    with open(os.path.join('templates', 'face_detail.html'), 'w', encoding='utf-8') as f:
        f.write(face_detail_html)

if __name__ == '__main__':
    # Tạo template nếu chưa có
    create_template_files()
    # Khởi tạo môi trường
    ensure_initialized()
    # Chạy app
    app.run(debug=True, host='0.0.0.0', port=5000)