import os
import io
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

# Initialiseer de app variabele die Gunicorn zoekt
app = Flask(__name__)

# Sta toe dat iedereen deze API mag aanroepen (nodig voor Elementor/WP)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- DE AI ENGINE ---
class LuxaGraphEngine:
    def __init__(self):
        # Initialiseer MediaPipe (Google AI)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def process(self, img_bytes, complexity=1000):
        # Zet bytes om naar beeld
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None
        
        height, width, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Gezichtsherkenning
        results = self.face_mesh.process(img_rgb)
        points = []

        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                for lm in fl.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])
        
        # 2. Achtergrond details (Randdetectie)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, int(complexity), 0.01, 10)
        if corners is not None:
            for i in np.int0(corners): points.append(i.ravel())

        # 3. Hoeken toevoegen (zodat het beeld vult)
        points.extend([[0, 0], [0, height-1], [width-1, 0], [width-1, height-1]])
        points = np.array(points)

        # 4. Triangulatie (Driehoeken maken)
        try:
            tri = Delaunay(points)
        except:
            return None

        # 5. Rendering (Inkleuren)
        output_img = np.zeros_like(img)

        for simplice in tri.simplices:
            pt1, pt2, pt3 = points[simplice[0]], points[simplice[1]], points[simplice[2]]
            triangle_cnt = np.array([pt1, pt2, pt3])
            
            # Bounding box optimalisatie
            rect = cv2.boundingRect(triangle_cnt)
            x, y, w, h = rect
            
            if x<0 or y<0 or (x+w)>width or (y+h)>height: continue
            
            roi = img[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            # Kleur bepalen (gemiddelde van de driehoek)
            avg_color = np.mean(roi, axis=(0, 1))
            color = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
            
            # Tekenen
            cv2.drawContours(output_img, [triangle_cnt], 0, color, -1, cv2.LINE_AA)

        # Terugsturen als afbeelding
        _, buffer = cv2.imencode('.jpg', output_img)
        return buffer.tobytes()

engine = LuxaGraphEngine()

# --- SERVER ENDPOINTS ---
@app.route('/', methods=['GET'])
def home():
    return "LuxaGraph AI Server is Live!"

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    # Lees complexiteit uit het formulier, standaard 1500
    complexity = float(request.form.get('complexity', 1500))
    
    processed_bytes = engine.process(file.read(), complexity)
    
    if processed_bytes:
        return send_file(
            io.BytesIO(processed_bytes),
            mimetype='image/jpeg'
        )
    return jsonify({"error": "Processing failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)