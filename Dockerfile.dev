FROM continuumio/anaconda3

RUN mkdir -p /opt/puddle
WORKDIR /opt/puddle

# Download dependencies.
ENV PATH "/opt/conda/bin:${PATH}"
RUN conda install -y nltk=3.4 nose coverage tqdm dill scikit-learn && conda clean --all -y
RUN conda install -c pytorch pytorch torchvision cudatoolkit=9.0
RUN pip install rednose frozendict pathos
ENV NOSE_REDNOSE 1

# Download pregex.
RUN mkdir pregex && cd pregex && curl -L https://github.com/insperatum/pregex/tarball/master | tar xz && mv $(find . -name "insperatum-pregex-*") pregex
ENV PYTHONPATH "/opt/puddle/pregex:${PYTHONPATH}"

# Download EC.
RUN curl -L https://github.com/ellisk42/ec/tarball/master | tar xz && mv $(find . -name "ellisk42-ec-*") ec
ENV PYTHONPATH "/opt/puddle/ec:${PYTHONPATH}"

# Download pyccg.
RUN curl -L https://github.com/hans/pyccg/tarball/master | tar xz && mv $(find . -name "hans-pyccg-*") pyccg
ENV PYTHONPATH "/opt/puddle/pyccg:${PYTHONPATH}"
