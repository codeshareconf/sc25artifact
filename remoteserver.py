from flask import Flask, request, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename
import os
import sys
import uuid
import json
import pyspdk

with open("settings.json", 'r') as f:
    settings = json.load(f)

FORMAT = settings["FORMAT"]
BLOCKSIZE = settings["BLOCKSIZE"]
DATADIR = settings["DATADIR"]
COPYFILEPREFIX = settings["COPYFILE_PREFIX"]
COUNT = 0

TRANSPORT_TYPE = 'TCP'
ADDRESS_FAMILY = 'IPv4'
TARGET_IP = '127.0.0.1'
TARGET_PORT = '4420'
NQN_NAME = 'nqn.2024-02.io.spdk:cnode1'

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return jsonify({"response": "true"})

@app.route('/readnvmefd', methods=['POST'])
def read_nvme():    
    json_data = request.json
    fname = json_data['fname']
    fsize = json_data['fsize']

    opfile = 'tmp/nvme_read_{}.{}'.format(uuid.uuid1().hex, FORMAT)
    buffer = pyspdk.fdread(fname.encode('utf-8'), fsize, BLOCKSIZE)
    with open(opfile, 'wb') as f:
        f.write(buffer)

    @after_this_request
    def remove_tempfile(response):
        try:
            os.remove(opfile)            
        except Exception as e:
            print("File cannot be deleted or not present")
        return response

    return(jsonify({'result':'success'}))

@app.route('/readnvmekv', methods=['POST'])
def read_nvme_lba():    
    json_data = request.json
    lba_list = json_data['lba_list']
    lba_count_list = json_data['lba_count_list'] 

    opfile = 'tmp/nvme_read_{}.{}'.format(uuid.uuid1().hex, FORMAT)    
    buffer = pyspdk.kvread(lba_list, lba_count_list, BLOCKSIZE)    
    with open(opfile, 'wb') as f:
        f.write(buffer)

    @after_this_request
    def remove_tempfile(response):
        try:
            os.remove(opfile)            
        except Exception as e:
            print("File cannot be deleted or not present")
        return response

    return(jsonify({'result':'success'}))

@app.route('/readhttp', methods=['POST'])
def read_http():
    if 'filename' not in request.files:
        print("No file part")
        return jsonify({"response": "Error - No file part"})
    file = request.files['filename']
    if file.filename == '':
        print('No selected file')
        return jsonify({"response": "Error - No selected file"})
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join('tmp',filename))

        return(jsonify({'result':'success'}))


if __name__ == "__main__":
    if sys.argv[1] == None:
        print("Port missing\n Correct Usage: python3 remote_server.py <port>")
    else:
        print(pyspdk.spdk_init(TRANSPORT_TYPE.encode('utf-8'), 
                                ADDRESS_FAMILY.encode('utf-8'), 
                                TARGET_IP.encode('utf-8'), 
                                TARGET_PORT.encode('utf-8'), 
                                NQN_NAME.encode('utf-8')))
        app.run(host="0.0.0.0", port=int(sys.argv[1]))    