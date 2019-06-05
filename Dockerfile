FROM pytorch/pytorch

RUN conda install scikit-learn pandas seaborn matplotlib -y
RUN pip install sacred addict gitpython pymongo pathos
