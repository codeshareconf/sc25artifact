from flask import Flask, request, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename
import os
import sys
import shutil
import uuid

app = Flask(__name__)

@app.route('/getfile/<classname>/<filename>')
def get_file(classname,filename):
    try:        
        fname = '/mnt/nvmedrive/datasets/imagedata/{}/{}'.format(classname, filename)        

        return send_file(fname, as_attachment=True, download_name=fname)
    except Exception as e:
        print(str(e))
        return "Error in file read"

if __name__ == "__main__":
    if sys.argv[1] == None:
        print("Port missing\n Correct Usage: python3 remote_server.py <port>")
    else:
        app.run(host="0.0.0.0", port=int(sys.argv[1]))