import os
import uuid
import traceback
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from vision.detector import detect_objects
from vision.analyzer import analyze_scene

app = Flask(__name__)

# -------------------------------
# BASE DIR (CRITICAL FIX)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# SAFE PATH CONFIG (ABSOLUTE PATHS)
# -------------------------------
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static/uploads')
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static/results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


# -------------------------------
# FILE VALIDATION
# -------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------
# HOME
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# -------------------------------
# UPLOAD + PROCESS
# -------------------------------
@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        try:
            # -----------------------
            # FILE CHECK
            # -----------------------
            if 'image' not in request.files:
                return render_template('upload.html', error='No file selected.')

            file = request.files['image']

            if file.filename == '':
                return render_template('upload.html', error='No file chosen.')

            if not allowed_file(file.filename):
                return render_template(
                    'upload.html',
                    error='Invalid file type. Upload PNG, JPG, JPEG, GIF, BMP.'
                )

            # -----------------------
            # SAVE FILE SAFELY
            # -----------------------
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"

            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(upload_path)

            # VERIFY FILE EXISTS (IMPORTANT)
            if not os.path.exists(upload_path):
                return render_template('upload.html', error='File upload failed on server.')

            # -----------------------
            # DETECTION PIPELINE
            # -----------------------
            result_filename, detected_objects = detect_objects(
                upload_path,
                app.config['RESULT_FOLDER']
            )

            # -----------------------
            # ANALYSIS PIPELINE
            # -----------------------
            scene_type, observations = analyze_scene(detected_objects)

            # -----------------------
            # REDIRECT RESULT PAGE
            # -----------------------
            return redirect(url_for(
                'result',
                result_img=result_filename,
                scene=scene_type,
                obs=','.join(observations),
                objects=','.join(detected_objects)
            ))

        except Exception as e:
            # FULL DEBUG LOGGING (CRITICAL FOR CLOUD)
            print("UPLOAD ERROR:", str(e))
            print(traceback.format_exc())

            return render_template(
                'upload.html',
                error='Internal server error during processing.'
            )

    return render_template('upload.html')


# -------------------------------
# RESULT PAGE
# -------------------------------
@app.route('/result')
def result():

    result_img = request.args.get('result_img', '')
    scene = request.args.get('scene', 'Unknown')

    obs_raw = request.args.get('obs', '')
    objects_raw = request.args.get('objects', '')

    observations = [o for o in obs_raw.split(',') if o]
    detected_objects = [o for o in objects_raw.split(',') if o]

    return render_template(
        'result.html',
        result_img=result_img,
        scene=scene,
        observations=observations,
        detected_objects=detected_objects
    )


# -------------------------------
# ABOUT PAGE
# -------------------------------
@app.route('/about')
def about():
    return render_template('about.html')


# -------------------------------
# MAIN ENTRY
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
