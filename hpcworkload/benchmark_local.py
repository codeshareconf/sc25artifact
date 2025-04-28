import os
import time
import threading

def read_file(filepath):
    with open(filepath, "rb") as f:
        _ = f.read()

def benchmark_reads(directory, num_threads=8):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
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

benchmark_reads("path_to_dataset_on_nvme_drive")