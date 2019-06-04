FROM pytorch/pytorch

COPY env.yaml .

RUN conda env create -f env.yaml
RUN activate dbmi && conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
