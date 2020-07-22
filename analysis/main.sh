# PARAMETER
# $1: (must) case name.  (ex) tf-pose
#
# PREREQUISITE
#   input file (must):  /home/ubuntu/input/sp02.mp4

# prepare 
case_name=$1
timestamp=$(date +%s)
analysis_home=/home/ubuntu/analysis
result_dir=${analysis_home}/result/${case_name}-${timestamp}

mkdir -p ${result_dir}

echo "TEST" ${result_dir}

${analysis_home}/nvidia-smi-loop.sh ${result_dir} &
pid=$!

# execute
nvidia-docker run -it --rm  \
    -v /home/ubuntu/input:/app/input \
    -v ${result_dir}:/app/output tf-pose-estimation:latest \
    python run_video.py --video=./input/baseball/sp02.mp4 --output=./output/${timestamp}-sp02-an.mp4

# afterword
kill ${pid}