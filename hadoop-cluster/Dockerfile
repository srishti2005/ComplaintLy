FROM apache/hadoop:3
USER root

ENV CONDA_DIR=/opt/conda
RUN curl -fsSL -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
 && rm /tmp/miniforge.sh

RUN $CONDA_DIR/bin/conda create -y -p /opt/pyenv python=3.11 pip \
      "pandas<2.2" "numpy<2" "pyarrow<16"

RUN /opt/pyenv/bin/pip install --no-cache-dir --default-timeout=120 --retries 10 pyspark==3.5.1

RUN $CONDA_DIR/bin/conda clean -afy \
 && chmod -R a+rX /opt/conda /opt/pyenv

ENV PATH=/opt/pyenv/bin:$PATH
ENV PYSPARK_PYTHON=/opt/pyenv/bin/python
USER hadoop