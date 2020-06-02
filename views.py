from flask import request, render_template, jsonify, send_from_directory
from classifier import classify
from app import app
import os


@app.route('/json', methods=['GET'])
def json():

    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        image_classes = classify(set(map(lambda file: os.path.join(app.config['UPLOAD_FOLDER'], file), files)))
        image_urls = list(map(lambda file: os.path.join(app.config['UPLOAD_FOLDER'], file), files))
        return jsonify(dict(zip(image_urls, image_classes)))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        images = request.files.getlist("file")
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            for file in files:
                os.remove(os.path.join(root, file))
        for image in images:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

    return render_template('index.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
