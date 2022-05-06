FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110:20220411

WORKDIR /stage

#Install huggingface transformers
RUN cd /stage && git clone https://github.com/jambayk/transformers.git &&\
    cd transformers &&\
    git checkout jambayk/ort_t5 &&\
    pip install -e .

# Install other dependencies
RUN pip install scikit-learn==1.0.2 datasets==2.0.0
