"""
Flask routes for image analysis API endpoints
"""

from flask import Blueprint, request, jsonify, send_file
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
import zipfile
from src.image_processor import ImageProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
image_analysis_bp = Blueprint('image_analysis', __name__)

# Configure upload settings
UPLOAD_FOLDER = '/tmp/uploads'
RESULTS_FOLDER = '/tmp/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@image_analysis_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Invalid Image Detection System",
        "version": "1.0.0"
    })

@image_analysis_bp.route('/upload-single', methods=['POST'])
def upload_single_image():
    """
    Upload and analyze a single image
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Initialize processor
        processor = ImageProcessor()
        
        # Process the image
        result = processor.process_single_image(file_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error processing single image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@image_analysis_bp.route('/upload-batch', methods=['POST'])
def upload_batch_images():
    """
    Upload and analyze multiple images
    """
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        # Create temporary directory for this batch
        batch_id = str(hash(str(files[0].filename) + str(len(files))))
        batch_upload_dir = os.path.join(UPLOAD_FOLDER, batch_id)
        batch_results_dir = os.path.join(RESULTS_FOLDER, batch_id)
        
        os.makedirs(batch_upload_dir, exist_ok=True)
        os.makedirs(batch_results_dir, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(batch_upload_dir, filename)
                file.save(file_path)
                saved_files.append(file_path)
        
        if not saved_files:
            return jsonify({"error": "No valid files to process"}), 400
        
        # Initialize processor
        processor = ImageProcessor()
        
        # Process the batch
        results = processor.process_directory(batch_upload_dir, batch_results_dir)
        
        # Create download links for result files
        result_files = {
            "json_report": f"/api/download-results/{batch_id}/image_analysis_report.json",
            "csv_summary": f"/api/download-results/{batch_id}/image_analysis_summary.csv",
            "statistics": f"/api/download-results/{batch_id}/statistics_summary.txt"
        }
        
        # Clean up uploaded files
        shutil.rmtree(batch_upload_dir)
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "results": results,
            "download_links": result_files
        })
        
    except Exception as e:
        logger.error(f"Error processing batch images: {str(e)}")
        return jsonify({"error": str(e)}), 500

@image_analysis_bp.route('/upload-zip', methods=['POST'])
def upload_zip_images():
    """
    Upload and analyze images from a ZIP file
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.zip'):
            return jsonify({"error": "File must be a ZIP archive"}), 400
        
        # Create temporary directories
        batch_id = str(hash(file.filename))
        zip_path = os.path.join(UPLOAD_FOLDER, f"{batch_id}.zip")
        extract_dir = os.path.join(UPLOAD_FOLDER, batch_id)
        batch_results_dir = os.path.join(RESULTS_FOLDER, batch_id)
        
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(batch_results_dir, exist_ok=True)
        
        # Save and extract ZIP file
        file.save(zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Initialize processor
        processor = ImageProcessor()
        
        # Process the extracted images
        results = processor.process_directory(extract_dir, batch_results_dir)
        
        # Create download links for result files
        result_files = {
            "json_report": f"/api/download-results/{batch_id}/image_analysis_report.json",
            "csv_summary": f"/api/download-results/{batch_id}/image_analysis_summary.csv",
            "statistics": f"/api/download-results/{batch_id}/statistics_summary.txt"
        }
        
        # Clean up uploaded files
        os.remove(zip_path)
        shutil.rmtree(extract_dir)
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "results": results,
            "download_links": result_files
        })
        
    except Exception as e:
        logger.error(f"Error processing ZIP file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@image_analysis_bp.route('/download-results/<batch_id>/<filename>', methods=['GET'])
def download_results(batch_id, filename):
    """
    Download result files
    """
    try:
        batch_results_dir = os.path.join(RESULTS_FOLDER, batch_id)
        file_path = os.path.join(batch_results_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@image_analysis_bp.route('/get-statistics/<batch_id>', methods=['GET'])
def get_statistics(batch_id):
    """
    Get processing statistics for a batch
    """
    try:
        batch_results_dir = os.path.join(RESULTS_FOLDER, batch_id)
        json_file = os.path.join(batch_results_dir, "image_analysis_report.json")
        
        if not os.path.exists(json_file):
            return jsonify({"error": "Results not found"}), 404
        
        import json
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        return jsonify({
            "success": True,
            "statistics": results.get("statistics", {}),
            "processing_info": results.get("processing_info", {})
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@image_analysis_bp.route('/cleanup/<batch_id>', methods=['DELETE'])
def cleanup_batch(batch_id):
    """
    Clean up result files for a batch
    """
    try:
        batch_results_dir = os.path.join(RESULTS_FOLDER, batch_id)
        
        if os.path.exists(batch_results_dir):
            shutil.rmtree(batch_results_dir)
            return jsonify({"success": True, "message": "Batch results cleaned up"})
        else:
            return jsonify({"error": "Batch not found"}), 404
        
    except Exception as e:
        logger.error(f"Error cleaning up batch: {str(e)}")
        return jsonify({"error": str(e)}), 500

@image_analysis_bp.route('/list-batches', methods=['GET'])
def list_batches():
    """
    List all available result batches
    """
    try:
        batches = []
        if os.path.exists(RESULTS_FOLDER):
            for batch_id in os.listdir(RESULTS_FOLDER):
                batch_dir = os.path.join(RESULTS_FOLDER, batch_id)
                if os.path.isdir(batch_dir):
                    # Get batch info from JSON file if available
                    json_file = os.path.join(batch_dir, "image_analysis_report.json")
                    batch_info = {"batch_id": batch_id}
                    
                    if os.path.exists(json_file):
                        import json
                        with open(json_file, 'r') as f:
                            results = json.load(f)
                            batch_info.update(results.get("processing_info", {}))
                    
                    batches.append(batch_info)
        
        return jsonify({
            "success": True,
            "batches": batches
        })
        
    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        return jsonify({"error": str(e)}), 500

