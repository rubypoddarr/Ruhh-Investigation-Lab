import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from vision.detector import detect_objects
from vision.analyzer import analyze_scene

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Landing page route."""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page — shows form on GET, processes image on POST."""
    if request.method == 'POST':
        # Check if a file was included in the request
        if 'image' not in request.files:
            return render_template('upload.html', error='No file selected.')

        file = request.files['image']

        if file.filename == '':
            return render_template('upload.html', error='No file chosen.')

        if not allowed_file(file.filename):
            return render_template('upload.html', error='Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP.')

        # Save the uploaded file with a unique name
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(upload_path)

        # Run object detection
        result_filename, detected_objects = detect_objects(upload_path, app.config['RESULT_FOLDER'])

        # Run scene analysis
        scene_type, observations = analyze_scene(detected_objects)

        return redirect(url_for('result',
                                result_img=result_filename,
                                scene=scene_type,
                                obs=','.join(observations),
                                objects=','.join(detected_objects)))

    return render_template('upload.html')


@app.route('/result')
def result():
    """Result page — shows the processed image and analysis."""
    result_img = request.args.get('result_img', '')
    scene = request.args.get('scene', 'Unknown')
    obs_raw = request.args.get('obs', '')
    objects_raw = request.args.get('objects', '')

    observations = [o for o in obs_raw.split(',') if o]
    detected_objects = [o for o in objects_raw.split(',') if o]

    return render_template('result.html',
                           result_img=result_img,
                           scene=scene,
                           observations=observations,
                           detected_objects=detected_objects)


@app.route('/about')
def about():
    """About page route."""
    return render_template('about.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
