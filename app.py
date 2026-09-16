import os
import json
import shutil
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
from dotenv import load_dotenv, set_key, dotenv_values
from PIL import Image
from pillow_heif import register_heif_opener
import fitz  # PyMuPDF

import process_images
import googlevision_translater
import note_translater
import export_responses
from pipeline_utils import atomic_write_json, INPUT_FOLDER

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

def _new_progress():
    return {
        'status': 'idle',
        'run_id': None,
        'total_groups': 0,
        'current_group': 0,
        'completed_groups': 0,
        'percentage': 0,
        'current_step': '',
        'groups': [],
    }


processing_progress = _new_progress()

PIPELINE = {
    'resize': process_images.process_group,
    'ocr': googlevision_translater.process_group,
    'transcribe': note_translater.process_group,
    'export': export_responses.export_run,
}

ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')

OUTPUT_FOLDER = os.path.expanduser(os.getenv('OUTPUT_FOLDER', '~/Desktop/markdown_output'))
TEMP_FOLDER = os.path.expanduser(os.getenv('TEMP_FOLDER', '/tmp/transcriber/temp_uploads'))
RUN_STATUS_FILE = os.getenv('RUN_STATUS_FILE', 'run_status.json')
CURRENT_RESPONSES_FILE = os.getenv('CURRENT_RESPONSES_FILE', 'responses_current.json')

# Create folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Clear temp folder on startup so stale uploads don't persist across restarts
if os.path.exists(TEMP_FOLDER):
    shutil.rmtree(TEMP_FOLDER)
os.makedirs(TEMP_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic', 'pdf'}
PDF_DPI = 150
PDF_MAX_PAGES = 100


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


def convert_pdf_to_images(file_path, original_filename):
    """Convert a PDF to a list of JPEG page dicts for upload_files() response.

    Each dict has: name, original_name, path, size, source_pdf.
    Raises ValueError for password-protected PDFs.
    Raises RuntimeError for zero-page or unopenable PDFs.
    """
    results = []
    secured_stem = os.path.splitext(os.path.basename(file_path))[0]

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError("PDF is password-protected and cannot be processed")

        page_count = min(doc.page_count, PDF_MAX_PAGES)
        if page_count == 0:
            raise RuntimeError("PDF contains no pages")

        if doc.page_count > PDF_MAX_PAGES:
            logger.warning(
                f"PDF '{original_filename}' has {doc.page_count} pages; "
                f"only the first {PDF_MAX_PAGES} will be extracted"
            )

        zoom = PDF_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)

            page_filename = f"{secured_stem}_p{page_num + 1:03d}.jpg"
            base = page_filename
            counter = 1
            while os.path.exists(os.path.join(TEMP_FOLDER, page_filename)):
                page_filename = f"{os.path.splitext(base)[0]}_{counter}.jpg"
                counter += 1

            page_path = os.path.join(TEMP_FOLDER, page_filename)
            pix.save(page_path, jpg_quality=85)

            results.append({
                'name': page_filename,
                'original_name': original_filename,
                'path': f'/api/temp/{page_filename}',
                'size': os.path.getsize(page_path),
                'source_pdf': os.path.basename(original_filename),
            })
    finally:
        doc.close()

    return results


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

                # Convert PDF to JPEG pages (1 PDF → N images)
                elif filename.lower().endswith('.pdf'):
                    try:
                        page_files = convert_pdf_to_images(file_path, original_filename)
                        os.remove(file_path)
                        if len(page_files) == 0:
                            errors.append(f'{original_filename}: PDF produced no images')
                            continue
                        uploaded_files.extend(page_files)
                        continue  # skip the normal .append() below
                    except ValueError as e:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        errors.append(f'{original_filename}: {e}')
                        continue
                    except Exception as e:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        errors.append(f'{original_filename}: PDF conversion error - {str(e)}')
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

        # Check file size before writing
        if len(request.data) == 0:
            return jsonify({'error': 'Empty file received'}), 400

        # Dedup: check if identical content already exists in temp folder
        import hashlib
        incoming_hash = hashlib.md5(request.data).hexdigest()
        for existing_file in os.listdir(TEMP_FOLDER):
            existing_path = os.path.join(TEMP_FOLDER, existing_file)
            if os.path.isfile(existing_path):
                with open(existing_path, 'rb') as ef:
                    if hashlib.md5(ef.read()).hexdigest() == incoming_hash:
                        logger.info(f"Duplicate upload ignored: {filename} matches existing {existing_file}")
                        return jsonify({'filename': existing_file, 'deduplicated': True}), 200

        # Write raw bytes to file
        with open(file_path, 'wb') as f:
            f.write(request.data)

        file_size = os.path.getsize(file_path)

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


TEMP_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'heic'}

@app.route('/api/temp', methods=['GET'])
def list_temp_files():
    """List image files currently in temp folder (PDFs excluded — they are transient)"""
    try:
        files = []
        for filename in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, filename)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if os.path.isfile(file_path) and ext in TEMP_IMAGE_EXTENSIONS:
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


def _persist_progress():
    atomic_write_json(RUN_STATUS_FILE, processing_progress)


def _update_group(group_name, **fields):
    for entry in processing_progress['groups']:
        if entry['name'] == group_name:
            entry.update(fields)
            break
    _persist_progress()


def _fail_group(group_name, error, failed_groups):
    logger.warning(f'Group {group_name} failed: {error}')
    _update_group(group_name, status='failed', error=error)
    failed_groups.append(group_name)


@app.route('/api/process', methods=['POST'])
def process_images_route():
    """Run the pipeline for every group in-process; a failed group never stops the run."""
    data = request.json or {}
    groups = data.get('groups', [])
    logger.info(f"Received processing request with {len(groups)} groups")
    if not groups:
        return jsonify({'error': 'No image groups provided'}), 400

    run_id = datetime.now().isoformat(timespec='seconds')

    # Start of run: clear last run's retry window and the input folder
    if os.path.exists(CURRENT_RESPONSES_FILE):
        os.remove(CURRENT_RESPONSES_FILE)
    if os.path.exists(INPUT_FOLDER):
        shutil.rmtree(INPUT_FOLDER)
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    processing_progress.clear()
    processing_progress.update(_new_progress())
    processing_progress.update({
        'status': 'processing', 'run_id': run_id, 'total_groups': len(groups),
        'current_step': 'Starting processing',
        'groups': [{
            'name': f"n{i+1}",
            'label': f"{len(g.get('images', []))} page{'s' if len(g.get('images', [])) != 1 else ''}",
            'status': 'pending', 'stage': None,
            'pages_done': 0, 'pages_total': len(g.get('images', [])),
            'attempts': 0, 'warnings': [], 'error': None,
        } for i, g in enumerate(groups)],
    })
    _persist_progress()

    for i, group in enumerate(groups):
        group_name = f"n{i+1}"
        group_folder = os.path.join(INPUT_FOLDER, group_name)
        os.makedirs(group_folder, exist_ok=True)
        images = group.get('images', [])
        for image_file in images:
            temp_path = os.path.join(TEMP_FOLDER, image_file)
            if os.path.exists(temp_path):
                shutil.copy2(temp_path, os.path.join(group_folder, image_file))
            else:
                logger.error(f"File not found in temp: {temp_path}")
        with open(os.path.join(group_folder, 'order.json'), 'w') as f:
            json.dump({'files': images}, f, indent=2)

    results, failed_groups = [], []
    completed = 0

    for i, group in enumerate(groups):
        group_name = f"n{i+1}"
        processing_progress.update({
            'current_group': i + 1,
            'current_step': f'Processing {group_name}',
            'percentage': int((i / len(groups)) * 100),
        })
        _update_group(group_name, status='running', stage='resizing')
        try:
            stage = PIPELINE['resize'](group_name, input_dir=INPUT_FOLDER)
            if not stage.ok:
                _fail_group(group_name, f'image processing: {stage.error}', failed_groups)
                continue

            _update_group(group_name, stage='ocr')
            stage = PIPELINE['ocr'](group_name, input_dir=INPUT_FOLDER)
            if not stage.ok:
                _fail_group(group_name, f'OCR: {stage.error}', failed_groups)
                continue

            result = PIPELINE['transcribe'](
                group_name, on_update=_update_group, input_dir=INPUT_FOLDER, run_id=run_id)
            results.append(result.to_dict())
            if result.status == 'failed':
                failed_groups.append(group_name)
            else:
                completed += 1
        except Exception as e:
            logger.exception(f'Unexpected error processing {group_name}')
            _fail_group(group_name, f'unexpected error: {e}', failed_groups)
        finally:
            processing_progress.update({
                'completed_groups': completed,
                'percentage': int(((i + 1) / len(groups)) * 100),
            })
            _persist_progress()

    if completed:
        processing_progress['current_step'] = 'Exporting results'
        _persist_progress()
        exportable = [r for r in results if r['status'] in ('done', 'warning')]
        for report in PIPELINE['export'](exportable, OUTPUT_FOLDER):
            if not report['ok']:
                completed -= 1
                _fail_group(report['group_name'], f"export: {report['error']}", failed_groups)

    processing_progress['current_step'] = 'Cleaning up temporary files'
    for filename in os.listdir(TEMP_FOLDER):
        file_path = os.path.join(TEMP_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    message = f'Processing completed: {completed}/{len(groups)} groups successful'
    if failed_groups:
        message += f', {len(failed_groups)} failed'
    processing_progress.update({
        'status': 'completed', 'current_step': 'Processing complete',
        'percentage': 100, 'completed_groups': completed,
    })
    _persist_progress()

    response_data = {
        'message': message,
        'groups_processed': completed,
        'total_groups': len(groups),
        'failed_groups': failed_groups,
    }
    if completed == 0:
        return jsonify(response_data), 500
    if failed_groups:
        return jsonify(response_data), 207
    return jsonify(response_data), 200


SENSITIVE_KEY_PATTERNS = ('_KEY', '_CREDENTIALS', '_SECRET', '_TOKEN', '_PASSWORD')


def is_sensitive(key):
    return any(key.upper().endswith(pat) for pat in SENSITIVE_KEY_PATTERNS)


@app.route('/api/config')
def get_config():
    """Get current configuration, with sensitive keys redacted."""
    config = dotenv_values(ENV_FILE)
    return jsonify({k: '***' if is_sensitive(k) else v for k, v in config.items()})


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration, rejecting sensitive keys."""
    data = request.json or {}
    rejected = [k for k in data if is_sensitive(k)]
    if rejected:
        return jsonify({'error': f'Cannot set sensitive keys via API: {rejected}'}), 400
    for key, value in data.items():
        set_key(ENV_FILE, key.upper(), str(value))
    load_dotenv(ENV_FILE, override=True)
    return jsonify({'message': 'Configuration updated'})


@app.route('/api/progress')
def get_progress():
    """Live progress, or the last run's saved status when idle."""
    if processing_progress.get('status') == 'idle' and os.path.exists(RUN_STATUS_FILE):
        try:
            with open(RUN_STATUS_FILE) as f:
                return jsonify(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    return jsonify(processing_progress)


if __name__ == '__main__':
    # app.run(debug=True, host='127.0.0.1', port=5001)
    app.run(debug=True, host='0.0.0.0', port=5001)
