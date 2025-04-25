import os
import cv2
import numpy as np
import sqlite3
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import uuid
from pathlib import Path
import argparse

# === CẤU HÌNH ===
DATA_ROOT = "data_files"
DB_PATH = os.path.join(DATA_ROOT, "face_recognition.db")
EMBEDDINGS_DIR = os.path.join(DATA_ROOT, "embeddings")
FACES_DIR = os.path.join(DATA_ROOT, "faces")

# Ngưỡng so sánh khuôn mặt
FACE_SIMILARITY_THRESHOLD = 0.7  # Cosine similarity threshold

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Khởi tạo thư mục ===
def init_directories():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)

# === Khởi tạo database ===
def init_database():
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
    
    conn.commit()
    conn.close()

# === Khởi tạo models ===
def init_models():
    # MTCNN cho việc phát hiện khuôn mặt
    mtcnn = MTCNN(
        image_size=160, margin=10, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.85,
        post_process=True, device=device
    )
    
    # FaceNet cho việc tạo embedding vector
    facenet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    
    return mtcnn, facenet

# === Trích xuất embedding từ khuôn mặt ===
def extract_face_embedding(face_img, mtcnn, facenet):
    try:
        # Sử dụng MTCNN để căn chỉnh khuôn mặt
        face = mtcnn(face_img)
        if face is None:
            return None
            
        # Chuyển face tensor sang cùng device với model FaceNet
        face = face.to(device)
            
        # Tạo embedding vector với FaceNet
        with torch.no_grad():
            embedding = facenet(face.unsqueeze(0)).detach().cpu().numpy()[0]
            
        # Kiểm tra NaN
        if np.isnan(embedding).any():
            print("Warning: Embedding contains NaN values")
            return None
            
        return embedding
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

# === Tìm khuôn mặt trùng khớp trong DB ===
def find_matching_face(embedding, threshold=FACE_SIMILARITY_THRESHOLD):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Lấy tất cả faceIds
    cursor.execute("SELECT id, embedding_path, user_id FROM faceIds")
    results = cursor.fetchall()
    
    best_match_id = None
    best_match_score = threshold
    best_match_user_id = None
    
    for face_id, embedding_path, user_id in results:
        try:
            # Load embedding từ file
            with open(embedding_path, 'rb') as f:
                stored_embedding = pickle.load(f)
            
            # Tính cosine similarity
            similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
            
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = face_id
                best_match_user_id = user_id
        except Exception as e:
            print(f"Error comparing with faceId {face_id}: {e}")
    
    conn.close()
    return best_match_id, best_match_user_id, best_match_score

# === Thêm khuôn mặt mới vào DB ===
def add_new_face(embedding, face_img):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Thêm vào bảng faceIds với user_id là NULL (chưa biết là ai)
    embedding_filename = f"embedding_{uuid.uuid4()}.pkl"
    embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_filename)
    
    # Lưu embedding
    with open(embedding_path, 'wb') as f:
        pickle.dump(embedding, f)
    
    # Thêm vào database
    cursor.execute(
        "INSERT INTO faceIds (embedding_path, user_id) VALUES (?, NULL)",
        (embedding_path,)
    )
    face_id = cursor.lastrowid
    
    # Tạo thư mục cho khuôn mặt này
    face_dir = os.path.join(FACES_DIR, f"face_{face_id}")
    os.makedirs(face_dir, exist_ok=True)
    
    # Lưu ảnh khuôn mặt
    face_img_path = os.path.join(face_dir, f"face_{uuid.uuid4()}.jpg")
    cv2.imwrite(face_img_path, face_img)
    
    conn.commit()
    conn.close()
    
    return face_id, face_dir

# === Gán khuôn mặt cho người dùng ===
def assign_face_to_user(face_id, user_id=None, user_name=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if user_id is None and user_name is not None:
        # Kiểm tra xem user_name đã tồn tại chưa
        cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
        result = cursor.fetchone()
        
        if result:
            user_id = result[0]
        else:
            # Tạo user mới
            cursor.execute("INSERT INTO users (name) VALUES (?)", (user_name,))
            user_id = cursor.lastrowid
    
    # Cập nhật faceId
    if user_id is not None:
        cursor.execute("UPDATE faceIds SET user_id = ? WHERE id = ?", (user_id, face_id))
    
    conn.commit()
    conn.close()
    
    return user_id

# === Xử lý khuôn mặt trong ảnh ===
def process_image(image_path, mtcnn, facenet):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return []
    
    # Chuyển sang RGB (MTCNN yêu cầu định dạng RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt bằng MTCNN
    boxes, _ = mtcnn.detect(img_rgb)
    
    results = []
    if boxes is None:
        print("No faces detected")
        return results
    
    # Xử lý từng khuôn mặt
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Cắt và lấy khuôn mặt
        face_img = img[y1:y2, x1:x2]
        
        # Trích xuất embedding
        embedding = extract_face_embedding(face_img, mtcnn, facenet)
        if embedding is None:
            continue
        
        # Tìm khuôn mặt trùng khớp
        face_id, user_id, similarity = find_matching_face(embedding)
        
        if face_id is not None:
            print(f"Found matching face ID: {face_id}, User ID: {user_id}, Similarity: {similarity:.2f}")
            
            # Lấy thông tin người dùng nếu có
            user_name = None
            if user_id is not None:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
                result = cursor.fetchone()
                if result:
                    user_name = result[0]
                conn.close()
            
            results.append({
                'face_id': face_id,
                'user_id': user_id,
                'user_name': user_name,
                'similarity': similarity,
                'bbox': (x1, y1, x2, y2),
                'embedding': embedding,
                'face_img': face_img
            })
        else:
            # Không tìm thấy khuôn mặt trùng khớp, thêm mới
            new_face_id, face_dir = add_new_face(embedding, face_img)
            print(f"New face detected! Added as face ID: {new_face_id}")
            
            results.append({
                'face_id': new_face_id,
                'user_id': None,
                'user_name': None,
                'similarity': 0.0,
                'bbox': (x1, y1, x2, y2),
                'embedding': embedding,
                'face_img': face_img,
                'is_new': True
            })
    
    return results

# === Tái học embeddings cho một người dùng ===
def relearn_user_embeddings(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Lấy tất cả face_ids của user_id
    cursor.execute("SELECT id FROM faceIds WHERE user_id = ?", (user_id,))
    face_ids = [row[0] for row in cursor.fetchall()]
    
    # Khởi tạo models
    mtcnn, facenet = init_models()
    
    for face_id in face_ids:
        # Lấy thư mục ảnh của face_id
        face_dir = os.path.join(FACES_DIR, f"face_{face_id}")
        if not os.path.exists(face_dir):
            continue
        
        # Lấy ảnh đầu tiên trong thư mục
        images = list(Path(face_dir).glob("*.jpg")) + list(Path(face_dir).glob("*.png"))
        if not images:
            continue
        
        # Xử lý ảnh đầu tiên
        img_path = str(images[0])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Trích xuất embedding mới
        embedding = extract_face_embedding(img, mtcnn, facenet)
        if embedding is None:
            continue
        
        # Cập nhật embedding trong database
        cursor.execute("SELECT embedding_path FROM faceIds WHERE id = ?", (face_id,))
        result = cursor.fetchone()
        if result:
            embedding_path = result[0]
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding, f)
    
    conn.commit()
    conn.close()
    print(f"Relearned embeddings for user_id {user_id}")

# === Gộp nhiều khuôn mặt thành một người ===
def merge_faces(face_ids, target_user_id=None, user_name=None):
    if not face_ids:
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tạo người dùng mới nếu cần
    if target_user_id is None and user_name is not None:
        cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
        result = cursor.fetchone()
        
        if result:
            target_user_id = result[0]
        else:
            cursor.execute("INSERT INTO users (name) VALUES (?)", (user_name,))
            target_user_id = cursor.lastrowid
    
    # Cập nhật tất cả face_ids thành target_user_id
    for face_id in face_ids:
        cursor.execute("UPDATE faceIds SET user_id = ? WHERE id = ?", (target_user_id, face_id))
    
    conn.commit()
    conn.close()
    
    return target_user_id

# === Tách một khuôn mặt sang ID mới ===
def split_face(face_id, new_user_name=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tạo người dùng mới nếu có tên
    new_user_id = None
    if new_user_name is not None:
        cursor.execute("INSERT INTO users (name) VALUES (?)", (new_user_name,))
        new_user_id = cursor.lastrowid
    
    # Cập nhật face_id
    cursor.execute("UPDATE faceIds SET user_id = ? WHERE id = ?", (new_user_id, face_id))
    
    conn.commit()
    conn.close()
    
    return new_user_id

# === Hiển thị khuôn mặt với bounding box và thông tin ===
def display_faces(image_path, results):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Vẽ bounding box và thông tin
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        user_id = result['user_id']
        face_id = result['face_id']
        user_name = result['user_name'] or "Unknown"
        
        # Màu cho bounding box (xanh cho người đã biết, đỏ cho người mới)
        color = (0, 255, 0) if user_id is not None else (0, 0, 255)
        
        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Hiển thị thông tin
        label = f"ID: {face_id}"
        if user_id is not None:
            label += f" ({user_name})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Hiển thị ảnh
    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === MAIN ===
def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--image', type=str, help='Path to the image')
    parser.add_argument('--assign', action='store_true', help='Assign user to face')
    parser.add_argument('--face_id', type=int, help='Face ID to assign')
    parser.add_argument('--user_id', type=int, help='User ID to assign')
    parser.add_argument('--user_name', type=str, help='User name to assign')
    parser.add_argument('--merge', action='store_true', help='Merge multiple faces into one user')
    parser.add_argument('--face_ids', type=str, help='Face IDs to merge (comma separated)')
    parser.add_argument('--relearn', action='store_true', help='Relearn embeddings for a user')
    
    args = parser.parse_args()
    
    # Khởi tạo
    init_directories()
    init_database()
    mtcnn, facenet = init_models()
    
    if args.image:
        # Xử lý ảnh
        results = process_image(args.image, mtcnn, facenet)
        display_faces(args.image, results)
    
    elif args.assign and args.face_id:
        # Gán face cho user
        user_id = assign_face_to_user(args.face_id, args.user_id, args.user_name)
        print(f"Assigned face ID {args.face_id} to user ID {user_id}")
    
    elif args.merge and args.face_ids:
        # Gộp nhiều faces
        face_ids = [int(x) for x in args.face_ids.split(',')]
        user_id = merge_faces(face_ids, args.user_id, args.user_name)
        print(f"Merged face IDs {face_ids} to user ID {user_id}")
    
    elif args.relearn and args.user_id:
        # Tái học embeddings
        relearn_user_embeddings(args.user_id)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
