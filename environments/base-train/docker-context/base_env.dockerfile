FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:20220329.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/hf-base

# Create conda environment
RUN conda create -y -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pip=20.2.4 

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install PyTorch
RUN pip install torch==1.11.0

WORKDIR /stage

#Install huggingface transformers and dependencies
RUN cd /stage && git clone https://github.com/jambayk/transformers.git &&\
    cd transformers &&\
    git checkout jambayk/ort_t5 &&\
    pip install -e .

# Install other dependencies
RUN pip install scikit-learn==1.0.2 datasets==2.0.0

# Install AzureML support
RUN pip install ruamel.yaml==0.17.6 --ignore-installed
RUN pip install azureml-core==1.35.0 azureml-mlflow==1.35.0

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
