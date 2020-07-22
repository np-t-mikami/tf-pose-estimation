FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y wget git swig vim \
    && apt-get install -y libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime

WORKDIR /app

#RUN git clone https://github.com/ildoonet/tf-pose-estimation.git
COPY . /app/


RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN cd /app/tf_pose/pafprocess/ \
    && swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace \
    && bash /app/models/graph/cmu/download.sh \
    && apt-get install -y libsm6 libxrender1 libxext-dev \
    && pip install opencv-python

# RUN python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg 