from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
import os
from inference import AnomalyDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector
# Ensure models are accessible using absolute path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'Models')
detector = AnomalyDetector(model_dir=MODEL_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':on 
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return render_template('index.html', filename=file.filename)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(detector.process_video(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_live')
def video_feed_live():
    # 0 is usually the default camera
    return Response(detector.process_video(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(detector.current_status)

if __name__ == '__main__':
    # Run slightly different port just in case, or default 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
