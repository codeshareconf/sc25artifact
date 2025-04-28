import os
import time
import threading
import pyspdk
import requests

def read_file(filepath):
    _ = requests.get('URL/{}'.format(filepath))

def benchmark_reads(directory, filelist, num_threads=8):
    files = filelist
    start_time = time.time()

    def worker(file_slice):
        for file in file_slice:
            read_file(file)
            # print(file)

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

filelist = get_file_list(2000)
benchmark_reads("path_to_dataset_on_nvme_drive", filelist)