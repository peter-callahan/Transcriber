import os
import json
import shutil
import logging
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF/HEIC format support
register_heif_opener()

# Load environment variables for Flask app
load_dotenv()

app = Flask(__name__)

# Configure logging for the Flask app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store processing progress
processing_progress = {
    'status': 'idle',
    'current_group': 0,
    'total_groups': 0,
    'current_step': '',
    'completed_groups': 0,
    'failed_groups': [],
    'percentage': 0
}

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

INPUT_FOLDER = os.path.expanduser(config['input_folder'])
OUTPUT_FOLDER = os.path.expanduser(config['output_folder'])
TEMP_FOLDER = os.path.expanduser(config['temp_folder'])

# Create folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_heic_to_jpeg(file_path):
    """Convert HEIC file to JPEG for browser compatibility"""
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if needed and save as JPEG
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Generate new filename with .jpg extension
            base_name = os.path.splitext(file_path)[0]
            jpeg_path = f"{base_name}.jpg"
            img.save(jpeg_path, "JPEG", quality=95)

            # Remove original HEIC file
            os.remove(file_path)
            return jpeg_path, True
    except Exception as e:
        logger.error(f"Error converting HEIC file {file_path}: {e}")
        return file_path, False


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
    """Upload files to temp folder with enhanced error handling"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        uploaded_files = []
        errors = []

        for file in files:
            try:
                if not file or file.filename == '':
                    errors.append(f'Empty file received')
                    continue

                if not allowed_file(file.filename):
                    errors.append(
                        f'{file.filename}: File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}')
                    continue

                filename = secure_filename(file.filename)
                if not filename:
                    errors.append(f'{file.filename}: Invalid filename')
                    continue

                # Handle duplicate filenames
                original_filename = filename
                counter = 1
                while os.path.exists(os.path.join(TEMP_FOLDER, filename)):
                    name, ext = os.path.splitext(original_filename)
                    filename = f"{name}_{counter}{ext}"
                    counter += 1

                file_path = os.path.join(TEMP_FOLDER, filename)

                # Save file with size check
                try:
                    file.save(file_path)

                    # Check file size after saving
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:
                        os.remove(file_path)
                        errors.append(f'{original_filename}: File is empty')
                        continue

                    # Check for reasonable file size (max 50MB)
                    max_size = 50 * 1024 * 1024  # 50MB
                    if file_size > max_size:
                        os.remove(file_path)
                        errors.append(
                            f'{original_filename}: File too large (max 50MB)')
                        continue

                except Exception as e:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    errors.append(
                        f'{original_filename}: Failed to save - {str(e)}')
                    continue

                # Convert HEIC to JPEG for browser compatibility
                if filename.lower().endswith('.heic'):
                    try:
                        converted_path, success = convert_heic_to_jpeg(
                            file_path)
                        if success:
                            filename = os.path.basename(converted_path)
                            file_path = converted_path
                        else:
                            errors.append(
                                f'{original_filename}: HEIC conversion failed')
                            continue
                    except Exception as e:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        errors.append(
                            f'{original_filename}: HEIC conversion error - {str(e)}')
                        continue

                uploaded_files.append({
                    'name': filename,
                    'original_name': original_filename,
                    'path': f'/api/temp/{filename}',
                    'size': os.path.getsize(file_path)
                })

            except Exception as e:
                errors.append(
                    f'{file.filename if hasattr(file, "filename") else "unknown"}: Unexpected error - {str(e)}')
                continue

        # Prepare response
        response_data = {
            'files': uploaded_files,
            'uploaded_count': len(uploaded_files),
            'error_count': len(errors)
        }

        if errors:
            response_data['errors'] = errors

        # Return appropriate status code
        if len(uploaded_files) == 0 and len(errors) > 0:
            return jsonify(response_data), 400
        elif len(errors) > 0:
            return jsonify(response_data), 207  # Partial success
        else:
            return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            'error': f'Server error during upload: {str(e)}',
            'files': [],
            'uploaded_count': 0,
            'error_count': 1
        }), 500


@app.route('/api/upload_raw', methods=['POST'])
def upload_raw_file():
    """Handle raw file upload from iOS Shortcuts"""
    try:
        if not request.data:
            return jsonify({'error': 'No data received'}), 400

        # Generate filename with timestamp
        import time
        timestamp = int(time.time() * 1000)

        print(request.headers)

        original_filename = request.headers.get('X-Filename', '')
        extension = request.headers.get('X-Extension', '')
        content_type = request.headers.get('Content-Type', '')

        # Determine extension from content type if not provided
        if not extension and content_type:
            if 'png' in content_type.lower():
                extension = '.png'
            elif 'jpeg' in content_type.lower() or 'jpg' in content_type.lower():
                extension = '.jpg'
            elif 'heic' in content_type.lower():
                extension = '.heic'

        if original_filename:
            # If extension is provided separately, append it
            if extension and not extension.startswith('.'):
                extension = f'.{extension}'

            # Check if filename already has an extension
            if extension and '.' not in os.path.basename(original_filename):
                filename = f"{original_filename}{extension}"
            else:
                filename = original_filename
        else:
            # Use detected extension or default to .jpg
            ext = extension if extension else '.jpg'
            filename = f"shortcut_{timestamp}{ext}"

        file_path = os.path.join(TEMP_FOLDER, filename)

        # Write raw bytes to file
        with open(file_path, 'wb') as f:
            f.write(request.data)

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            os.remove(file_path)
            return jsonify({'error': 'Empty file received'}), 400

        # Verify the file is a valid image and detect actual format
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                actual_format = img.format
                detected_ext = f".{actual_format.lower()}" if actual_format else None

                # Warn if extension doesn't match actual format
                if detected_ext and not filename.lower().endswith(detected_ext):
                    logger.warning(f"File extension mismatch: {filename} is actually {actual_format}")
                    logger.info(f"Content-Type: {content_type}, Detected: {actual_format}, Extension: {extension}")
        except Exception as e:
            logger.error(f"Failed to verify image format for {filename}: {e}")
            os.remove(file_path)
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

        logger.info(f"Raw upload received: {filename}, size: {file_size}, format: {actual_format if 'actual_format' in locals() else 'unknown'}")

        return jsonify({
            'files': [{
                'name': filename,
                'original_name': filename,
                'path': f'/api/temp/{filename}',
                'size': file_size
            }],
            'uploaded_count': 1,
            'error_count': 0
        })

    except Exception as e:
        logger.error(f"Raw upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/temp/<filename>')
def get_temp_image(filename):
    """Serve temp images"""
    return send_file(os.path.join(TEMP_FOLDER, filename))


@app.route('/api/temp', methods=['GET'])
def list_temp_files():
    """List all files currently in temp folder"""
    try:
        files = []
        for filename in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, filename)
            if os.path.isfile(file_path):
                files.append({
                    'name': filename,
                    'original_name': filename,
                    'path': f'/api/temp/{filename}',
                    'size': os.path.getsize(file_path)
                })

        files.sort(key=lambda x: x['name'])

        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/processed/<group_name>/<filename>')
def get_processed_image(group_name, filename):
    """Serve processed images from input folder (what APIs actually see)"""
    if group_name == 'temp':
        # For ungrouped images, serve from temp folder (this is what would be processed)
        return send_file(os.path.join(TEMP_FOLDER, filename))

    processed_path = os.path.join(INPUT_FOLDER, group_name, filename)
    if os.path.exists(processed_path):
        return send_file(processed_path)
    else:
        # Fallback to temp image if processed version doesn't exist yet
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
        logger.info(f"Received processing request with data: {data}")
        groups = data.get('groups', [])
        logger.info(f"Extracted {len(groups)} groups from request")

        for i, group in enumerate(groups):
            logger.info(f"Group {i}: {group}")

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

            images_in_group = group.get('images', [])
            logger.info(f"Group {group_name} has {len(images_in_group)} images: {images_in_group}")

            for image_file in images_in_group:
                temp_path = os.path.join(TEMP_FOLDER, image_file)
                new_path = os.path.join(group_folder, image_file)
                logger.info(f"Attempting to copy {temp_path} to {new_path}")
                if os.path.exists(temp_path):
                    shutil.copy2(temp_path, new_path)
                    logger.info(f"Successfully copied {image_file}")
                else:
                    logger.error(f"File not found in temp: {temp_path}")

            # Write order.json to preserve UI order
            order_file = os.path.join(group_folder, 'order.json')
            with open(order_file, 'w') as f:
                json.dump({'files': images_in_group}, f, indent=2)
            logger.info(f"Wrote file order to {order_file}: {images_in_group}")

        # Step 3: Process each group sequentially
        global processing_progress
        completed_groups = 0
        failed_groups = []

        # Initialize progress tracking
        processing_progress.update({
            'status': 'processing',
            'total_groups': len(groups),
            'current_group': 0,
            'current_step': 'Starting processing',
            'completed_groups': 0,
            'failed_groups': [],
            'percentage': 0
        })

        for i, group in enumerate(groups):
            group_name = f"n{i+1}"
            logger.info(f"Processing group {group_name} ({i+1}/{len(groups)})")

            # Update progress
            processing_progress.update({
                'current_group': i + 1,
                'current_step': f'Processing group {group_name}',
                'percentage': int((i / len(groups)) * 100)
            })

            try:
                # Resize images for this group
                processing_progress['current_step'] = f'Resizing images for {group_name}'
                result = subprocess.run(['python', 'process_images.py', group_name],
                                        cwd=os.getcwd(), capture_output=True, text=True)
                logger.info(f'process_images.py STDOUT: {result.stdout}')
                logger.info(f'process_images.py STDERR: {result.stderr}')
                if result.returncode != 0:
                    logger.warning(f'Image processing failed for {group_name}')
                    failed_groups.append(f'{group_name} (image processing)')
                    continue

                # OCR with Google Vision for this group
                processing_progress['current_step'] = f'Running OCR for {group_name}'
                env = os.environ.copy()
                env['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv(
                    'GOOGLE_APPLICATION_CREDENTIALS')
                result = subprocess.run(['python', 'googlevision-translater.py', group_name],
                                        cwd=os.getcwd(), env=env, capture_output=True, text=True)
                logger.info(f'googlevision-translater.py STDOUT: {result.stdout}')
                logger.info(f'googlevision-translater.py STDERR: {result.stderr}')
                if result.returncode != 0:
                    logger.warning(f'OCR processing failed for {group_name}')
                    failed_groups.append(f'{group_name} (OCR)')
                    continue

                # Text conversion with GPT-4 for this group
                processing_progress['current_step'] = f'Converting text for {group_name}'
                env = os.environ.copy()
                openai_key = os.getenv('OPENAI_API_KEY')
                env['OPENAI_API_KEY'] = openai_key
                result = subprocess.run(['python', 'gpt4-note-translater.py', group_name],
                                        cwd=os.getcwd(), env=env, capture_output=True, text=True)
                logger.info(f'gpt4-note-translater.py STDOUT: {result.stdout}')
                logger.info(f'gpt4-note-translater.py STDERR: {result.stderr}')
                if result.returncode != 0:
                    logger.warning(f'Text conversion failed for {group_name}')
                    failed_groups.append(f'{group_name} (text conversion)')
                    continue

                completed_groups += 1
                logger.info(f"Successfully completed group {group_name}")

                # Update progress after successful completion
                processing_progress.update({
                    'completed_groups': completed_groups,
                    'current_step': f'Completed {group_name}',
                    'percentage': int(((i + 1) / len(groups)) * 100)
                })

            except Exception as e:
                logger.error(f'Error processing group {group_name}: {str(e)}')
                failed_groups.append(f'{group_name} (error: {str(e)})')
                continue

        # Export all completed responses at the end
        if completed_groups > 0:
            processing_progress['current_step'] = 'Exporting results'
            result = subprocess.run(['python', 'export_responses.py'],
                                    cwd=os.getcwd())
            if result.returncode != 0:
                logger.warning('Export failed')

        # Step 4: Clear temp files after successful processing
        processing_progress['current_step'] = 'Cleaning up temporary files'
        for filename in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Prepare response message
        message = f'Processing completed: {completed_groups}/{len(groups)} groups successful'
        if failed_groups:
            message += f', {len(failed_groups)} failed'

        # Update final progress status
        processing_progress.update({
            'status': 'completed',
            'current_step': 'Processing complete',
            'percentage': 100,
            'completed_groups': completed_groups,
            'failed_groups': failed_groups
        })

        response_data = {
            'message': message,
            'groups_processed': completed_groups,
            'total_groups': len(groups),
            'failed_groups': failed_groups
        }

        # Return appropriate status code
        if completed_groups == 0:
            return jsonify(response_data), 500
        elif failed_groups:
            return jsonify(response_data), 207  # Partial success
        else:
            return jsonify(response_data), 200

    except Exception as e:
        # Update progress on error
        processing_progress.update({
            'status': 'error',
            'current_step': f'Error: {str(e)}',
            'percentage': 0
        })
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


@app.route('/api/progress')
def get_progress():
    """Get current processing progress"""
    return jsonify(processing_progress)


if __name__ == '__main__':
    # app.run(debug=True, host='127.0.0.1', port=5001)
    app.run(debug=True, host='0.0.0.0', port=5001)
