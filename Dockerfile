FROM python:3.11

#working directory

WORKDIR /repo

# copy in the directorycam_lid\Scripts\activate
COPY . /repo

# install requirements

RUN pip install -r requirements.txt
