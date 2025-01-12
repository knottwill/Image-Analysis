# use the Miniconda3 base image
FROM continuumio/miniconda3

# install git and curl
# (final line removes package lists which are no longer needed)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# set the working directory in the Docker image
WORKDIR /project

# copy the current directory contents into the container at /project
COPY . /project

# update the conda environment
RUN conda env update --file env_very_light.yml --name base
RUN pip install pywavelets