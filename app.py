import os
import json
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables for Flask app
load_dotenv()

app = Flask(__name__)

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

INPUT_FOLDER = config['input_folder']
OUTPUT_FOLDER = config['output_folder']
TEMP_FOLDER = config['temp_folder']

# Create folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/groups')
def get_groups():
    """Get saved groups from session storage (placeholder - could be saved to file)"""
    # For now, just return empty - groups will be managed client-side
    return jsonify([])


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload files to temp folder"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    uploaded_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(TEMP_FOLDER, filename)
            file.save(file_path)
            uploaded_files.append({
                'name': filename,
                'path': f'/api/temp/{filename}'
            })

    return jsonify({'files': uploaded_files})


@app.route('/api/temp/<filename>')
def get_temp_image(filename):
    """Serve temp images"""
    return send_file(os.path.join(TEMP_FOLDER, filename))


@app.route('/api/clear_temp', methods=['POST'])
def clear_temp():
    """Clear all files from temp folder"""
    try:
        cleared_files = []
        for filename in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                cleared_files.append(filename)

        return jsonify({
            'message': f'Cleared {len(cleared_files)} temporary files',
            'cleared_files': cleared_files
        })
    except Exception as e:
        return jsonify({'error': f'Failed to clear temp files: {str(e)}'}), 500


@app.route('/api/process', methods=['POST'])
def process_images():
    """Run the complete processing pipeline with groups"""
    try:
        data = request.json
        groups = data.get('groups', [])

        if not groups:
            return jsonify({'error': 'No image groups provided'}), 400

        # Step 1: Clear existing input folders
        if os.path.exists(INPUT_FOLDER):
            shutil.rmtree(INPUT_FOLDER)
        os.makedirs(INPUT_FOLDER, exist_ok=True)

        # Step 2: Create folders for each group and move images
        for i, group in enumerate(groups):
            group_name = f"n{i+1}"
            group_folder = os.path.join(INPUT_FOLDER, group_name)
            os.makedirs(group_folder, exist_ok=True)

            for image_file in group.get('images', []):
                temp_path = os.path.join(TEMP_FOLDER, image_file)
                new_path = os.path.join(group_folder, image_file)
                if os.path.exists(temp_path):
                    # Copy instead of move to keep temp files
                    shutil.copy2(temp_path, new_path)

        # Step 3: Run existing processing scripts
        # Resize images
        result = subprocess.run(['python', 'process_images.py'],
                                capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            return jsonify({'error': f'Image processing failed: {result.stderr}'}), 500

        # OCR with Google Vision
        env = os.environ.copy()
        env['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv(
            'GOOGLE_APPLICATION_CREDENTIALS')
        result = subprocess.run(['python', 'googlevision-translater.py'],
                                capture_output=True, text=True, cwd=os.getcwd(), env=env)
        if result.returncode != 0:
            return jsonify({'error': f'OCR processing failed: {result.stderr}'}), 500

        # Text conversion with GPT-4
        env = os.environ.copy()
        # Use the project key instead of service account key
        openai_key = os.getenv('OPENAI_API_KEY')
        print(
            f"DEBUG: Flask loaded OpenAI key: {openai_key[:20] + '...' if openai_key else 'None'}")
        env['OPENAI_API_KEY'] = openai_key
        result = subprocess.run(['python', 'gpt4-note-translater.py'],
                                capture_output=True, text=True, cwd=os.getcwd(), env=env)
        if result.returncode != 0:
            return jsonify({'error': f'Text conversion failed: {result.stderr}'}), 500

        # Export responses (optional final step)
        result = subprocess.run(['python', 'export_responses.py'],
                                capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            # Don't fail the whole process
            print(f'Warning: Export failed: {result.stderr}')

        # Step 4: Clear temp files after successful processing
        for filename in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return jsonify({
            'message': f'Processing completed successfully for {len(groups)} groups',
            'groups_processed': len(groups)
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/config')
def get_config():
    """Get current configuration"""
    return jsonify(config)


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    global config
    data = request.json
    config.update(data)

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return jsonify({'message': 'Configuration updated'})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
