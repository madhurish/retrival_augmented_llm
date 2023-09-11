from flask import Flask, request, jsonify, send_from_directory
from qallm import LLM_PDF_QA
import os

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

qa_instance = LLM_PDF_QA()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        docs = qa_instance.load_from_directory(filepath)
        qa_instance.save_db(docs)
        return jsonify({'success': True, 'message': 'File uploaded and DB updated successfully'}), 200

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/ask', methods=['POST'])
def ask_query():
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query'}), 400

    response = qa_instance.run(data['query'])
    return jsonify({'response': response})

@app.route('/')
def index():
    return '<h1>Hello!</h1>'


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
