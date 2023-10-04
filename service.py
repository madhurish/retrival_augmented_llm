from flask import Flask, request, jsonify, send_from_directory
from qallm import LLM_PDF_QA
import os

app = Flask(__name__)

UPLOAD_FOLDER = './docs'
CHAT_FOLDER = './chats'
ALLOWED_EXTENSIONS = {'pdf','txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CHAT_FOLDER'] = CHAT_FOLDER

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
        # qa_instance.delete_files_in_directory(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        docs = qa_instance.load_from_directory(app.config['UPLOAD_FOLDER'])
        qa_instance.save_db(docs)
        return jsonify({'success': True, 'message': 'File uploaded and DB updated successfully'}), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/upload_chat',methods=['POST'])
def upload_chat():
    if 'chat' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    chat = request.files['chat']
    if chat.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if chat and allowed_file(chat.filename):
        chatpath = os.path.join(app.config['CHAT_FOLDER'],chat.filename)
        chat.save(chatpath)
        return jsonify({'success': True, 'message': 'File uploaded and DB updated successfully'}), 200


@app.route('/list_files', methods=['GET'])
def list_files():
    files=[f for f in os.listdir(app.config['UPLOAD_FOLDER'])]
    return jsonify({'files':files})


@app.route('/list_chats', methods=['GET'])
def list_chats():
    files=[f for f in os.listdir(app.config['CHAT_FOLDER'])]
    return jsonify({'chats':files})


@app.route('/delete_chats',methods=['POST'])
def delete_chats():
    data= request.json
    if not data or 'chat_name' not in data:
        return jsonify({'error':'Provide file name'}), 400
    status = qa_instance.delete_file_in_directory(app.config['CHAT_FOLDER'],data['chat_name'])
    if status==400:
        return jsonify({'error':'could not delete chat'}), 400
    else:
        return jsonify({'success': True, 'message': 'Chat deleted successfully'}), 200
    

@app.route('/delete_file', methods=['POST'])
def delete_file():
    data=request.json
    if not data or 'file_name' not in data:
        return jsonify({'error':'Provide file name'}), 400
    status= qa_instance.delete_file_in_directory(app.config['UPLOAD_FOLDER'],data['file_name'])
    if status==400:
        return jsonify({'error':'could not delete file'}), 400
    else:
        docs = qa_instance.load_from_directory(app.config['UPLOAD_FOLDER'])
        if docs or docs!=[]:
            qa_instance.save_db(docs)
            return jsonify({'success': True, 'message': 'File deleted and DB updated successfully'}), 200
        return jsonify({'success': False, 'message': 'File deleted but DB not updated as some context is needed in DB'}), 200
    
@app.route('/ask', methods=['POST'])
def ask_query():
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query'}), 400

    response = qa_instance.run(data['query'])
    return jsonify({'response': response})

@app.route('/classify', methods=['POST'])
def classify_query():
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query'}), 400

    response = qa_instance.classify(data['query'])
    return jsonify({'response': response})

@app.route('/no_context_ask', methods=['POST'])
def ask_query_no_context():
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query'}), 400

    response = qa_instance.answer(data['query'])
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
