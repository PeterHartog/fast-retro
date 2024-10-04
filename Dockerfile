FROM continuumio/miniconda3

WORKDIR /app

ENV PYTHONPATH="/app/seldon_deploy:${PYTHONPATH}"

SHELL ["/bin/bash", "--login", "-c"]

COPY . /app

RUN conda env create -f seldon_deploy/environment.yaml && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy
RUN echo "conda activate chemformer && export LD_LIBRARY_PATH=\$(conda run -n chemformer echo \${CONDA_PREFIX}/lib)" > ~/.bashrc


EXPOSE 5000 6000 8000 9000

# Define environment variable
ENV MODEL_NAME={model_name} SERVICE_TYPE=MODEL PERSISTENCE=0

ENTRYPOINT []
CMD exec seldon-core-microservice $MODEL_NAME \
    --service-type $SERVICE_TYPE \
    --persistence $PERSISTENCE \
    --http-port 5000 \
    --grpc-port 9000
