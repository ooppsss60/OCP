from flask import Flask
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'
import views

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
