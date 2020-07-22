output_dir=$1

log_file=${output_dir}/nvidia-smi.log
query=timestamp,memory.total,memory.used,memory.free,utilization.memory,utilization.gpu


nvidia-smi --query-gpu=${query} --format=csv,nounits > ${log_file}
while true; do
    nvidia-smi --query-gpu=${query} --format=csv,noheader,nounits >> ${log_file}
    sleep 1
done