# SC25 Artifacts
This project is used for benchmarking NVMe-oF-FD.

## Setup

### Setting up the Target NVMe Server
Use the following steps to setup a Target NVMe server. This is the server where the benchmarking data is stored and the benchmarking experiments are run from.

First set the environment variable
```bash
ipaddrserver=10.0.0.0                #Ip address of the Target NVMe Server
app_port=4421                        #Port on which NVMe-oTCP connections will be made
nqnname="nqn.2024-02.io.spdk:cnode1" #Namespace used for the target
```

Use the following steps to bring up the Target NVMe Server
```bash
git clone https://github.com/spdk/spdk.git
cd spdk
sudo HUGEMEM=32768 ./scripts/setup.sh #Set the HUGEMEM as per your needs
sudo nohup ./build/bin/nvmf_tgt --wait-for-rpc & sleep 2
sudo ./scripts/rpc.py bdev_nvme_set_options -t 0 -a none -p 100000
sudo ./scripts/rpc.py sock_impl_set_options --enable-zerocopy-send-server --enablezerocopy-send-client -p 2 -i posix #You can skip if this command is not working
sudo ./scripts/rpc.py framework_start_init
sudo ./scripts/rpc.py nvmf_create_transport -t tcp -q 128 -c 4096 -i 131072 -u 131072 -a 128 -b 32 -n 4096
sudo ./scripts/rpc.py nvmf_create_subsystem ${nqnname}
sudo ./scripts/rpc.py nvmf_subsystem_allow_any_host -e ${nqnname}
sudo ./scripts/rpc.py nvmf_subsystem_add_listener -t tcp -a ${ipaddrserver} -s ${app_port} ${nqnname}
sudo ./scripts/rpc.py bdev_aio_create /dev/nvme2n1 aio0 #Replace the path as per your lsblk output on your system
sudo ./scripts/rpc.py nvmf_subsystem_add_ns ${nqnname} aio0
```

Use the following steps to see if the target server is set.
```bash
sudo ./scripts/rpc.py nvmf_get_subsystems
sudo ./scripts/rpc.py nvmf_get_transports
```

### Setting up the intiator
Use the following steps to setup the intiator where pyspdk will run.
```bash
git clone https://github.com/spdk/spdk.git
cd spdk
sudo HUGEMEM=32768 ./scripts/setup.sh
```

### Settings file
Create a `settings.json` file which will be used for the runs. The contents are as follows.
```json
{
    "NVMEFD_LATENCY_DIR" : "pathtodir",
    "NVMEKV_LATENCY_DIR" : "pathtodir",
    "HTTP_LATENCY_DIR" : "pathtodir",
    "GRPC_LATENCY_DIR" : "pathtodir",
    "DATA_DIR" : "/mnt/nvmedrive/video_benchmark_data/",
    "COPYFILE_PREFIX" : "activity_test_",
    "BASE_FILE" : "/mnt/nvmedrive/activity_test_1.mp4",
    "FORMAT" : "mp4",
    "BLOCKSIZE" : 512,
    "GRPC_SERVER_URL": "URL",
    "HTTP_SERVER_URL": "URL/readhttp",
    "NVMEFD_SERVER_URL": "URL/readnvmefd",
    "NVMEKV_SERVER_URL": "URL/readnvmekv"
}
```

### Running the experiments
Use the following steps to setup each of the different approaches.

#### NVMe-oF-FD and NVMe-oF-KV
1. Build the pyspdk library to get the `pyspsdk.so` file or get it from the pyspdk repo.
2. Run `remoteserver.py` on the initator.
3. Run `nvmeoffd_bench.py` or `nvmeofkv_bench.py` on the target.

#### HTTP
2. Run `remoteserver.py` on the initator.
3. Run `http_bench.py` on the target.

#### GRPC
2. Run `grpc_server.py` on the initator.
3. Run `grpc_bench.py` on the target.

All results will be stored in the directories set in the `settings.json` file.
