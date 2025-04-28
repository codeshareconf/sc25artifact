import requests
import time
import numpy as np
import os
from collections import defaultdict
from pathlib import Path
import json

settings = {}

with open("settings.json", 'r') as f:
    settings = json.load(f)

LATENCYDIR = settings["NVMEFD_LATENCY_DIR"]
DATADIR = settings["DATA_DIR"]
COPYFILEPREFIX = settings["COPYFILE_PREFIX"]
FORMAT = settings["FORMAT"]
URL = settings["NVMEKV_SERVER_URL"]

def run(run_number, limit):

    opfile = open('{}nvmeoffd_latency_{}.txt'.format(LATENCYDIR, run_number), 'w')
    
    for j in range(limit):
        timelist = []
        ftlist = []
        for i in range(2**j):            
            ft = time.time()
            dest = '{}{}{}.{}'.format(DATADIR, COPYFILEPREFIX, i+1, FORMAT)
            nvme_dict = {
                "fname": dest,
                "fsize": os.path.getsize(dest)
            }
            ftlist.append(time.time() - ft)
            t1 = time.time()
            requests.post(URL, json=nvme_dict)
            timelist.append(time.time() - t1)         
        opstr = '{}\t{}\t{}\t{}\t{}\n'.format(2**j, np.sum(timelist), np.mean(timelist), np.sum(ftlist), np.mean(ftlist))
        opfile.write(opstr)

    opfile.close()

if __name__ == '__main__':
    nrofruns = 1
    LIMIT = 11
    
    if not os.path.exists(LATENCYDIR):
        Path(LATENCYDIR).mkdir(parents=True, exist_ok=True)

    for i in range(nrofruns):
        run(i+1, LIMIT)
    
    latency_file = open('{}fulllatency_{}.txt'.format(LATENCYDIR, nrofruns),'w')
    tdict = defaultdict(list)
    ftdict = defaultdict(list)
    for f in os.listdir(LATENCYDIR):
        if len(f.strip().split('_')) > 2:
            ip = open(LATENCYDIR+f)

            for eachline in ip:
                data = eachline.strip().split('\t')                
                tdict[int(data[0])].append([float(data[1])])
                ftdict[int(data[0])].append([float(data[3])])
    
    for k in sorted(tdict.keys()):
        opstr = '{}\t{}\t{}\n'.format(k, str(np.mean(tdict[k])), str(np.mean(ftdict[k])))
        latency_file.write(opstr)