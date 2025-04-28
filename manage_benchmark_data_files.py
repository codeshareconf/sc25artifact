import shutil
import sys
from subprocess import Popen
from random import randint
from pathlib import Path
import json
import os

with open("settings.json", 'r') as f:
    settings = json.load(f)

DATADIR = settings["DATA_DIR"]
COPYFILEPREFIX = settings["COPYFILE_PREFIX"]
BASEFILE = settings["BASE_FILE"]
FORMAT = settings["FORMAT"]

def create():
    if not os.path.exists(DATADIR):
        Path(DATADIR).mkdir(parents=True, exist_ok=True)

    for i in range(5000):
        dest = '{}{}{}.{}'.format(DATADIR,COPYFILEPREFIX,i+1,FORMAT)
        shutil.copy(BASEFILE, dest)

def uncache():
    for i in range(5000):
        dest = '{}{}{}.{}'.format(DATADIR,COPYFILEPREFIX,i+1,FORMAT)
        cmd = "./dbsake uncache {}".format(dest)
        Popen(cmd, shell=True)

def check():
    flist = []
    for i in range(10):
        r = randint(1,5000)
        if r not in flist:
            flist.append(r)

    for f in flist:
        dest = '{}{}{}.{}'.format(DATADIR,COPYFILEPREFIX,f,FORMAT)
        cmd = "./dbsake fincore {}".format(dest)
        Popen(cmd, shell=True)
if __name__ == '__main__':
    
    op = int(sys.argv[1])

    if op == 0:
        create()
    if op == 1:
        uncache()
    elif op == 2:
        check()