# monitor.sh
#!/bin/bash
echo "Timestamp,GPU Utilization (%),I/O Wait (%)" > metrics.csv
while true; do
  timestamp=$(date +%s)
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  io_wait=$(iostat -c 1 2 | grep -A1 "%iowait" | tail -n1 | awk '{print $4}')
  echo "$timestamp,$gpu_util,$io_wait" >> metrics.csv
  sleep 1
done