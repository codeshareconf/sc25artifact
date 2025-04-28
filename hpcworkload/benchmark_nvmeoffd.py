import os
import time
import threading
import pyspdk

TRANSPORT_TYPE = 'TCP'
ADDRESS_FAMILY = 'IPv4'
TARGET_IP = '127.0.0.1'
TARGET_PORT = '4420'
NQN_NAME = 'nqn.2024-02.io.spdk:cnode1'

def read_file(filepath):
    _ = pyspdk.fdread(filepath.encode('utf-8'), 4096)

def benchmark_reads(directory, filelist, num_threads=8):
    files = [os.path.join(directory, f) for f in filelist]
    start_time = time.time()

    def worker(file_slice):
        for file in file_slice:
            read_file(file)

    # Split files across threads
    chunk_size = len(files) // num_threads
    threads = []
    for i in range(num_threads):
        start = i * chunk_size
        end = None if i == num_threads - 1 else (i + 1) * chunk_size
        t = threading.Thread(target=worker, args=(files[start:end],))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    elapsed = end_time - start_time
    total_data_mb = len(files) * 0.5  # Assuming 512KB per file
    throughput = total_data_mb / elapsed
    print(f"Directory: {directory} | Time: {elapsed:.2f}s | Throughput: {throughput:.2f} MB/s")


def get_file_list(num_files = 2000):
    flist = []

    for i in range(num_files):
        flist.append('file_{}.bin'.format(i))
    return flist

print(pyspdk.spdk_init(TRANSPORT_TYPE.encode('utf-8'), 
                            ADDRESS_FAMILY.encode('utf-8'), 
                            TARGET_IP.encode('utf-8'), 
                            TARGET_PORT.encode('utf-8'), 
                            NQN_NAME.encode('utf-8')))
filelist = get_file_list(2000)
benchmark_reads("path_to_dataset_on_nvme_drive", filelist)