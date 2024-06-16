FROM python:3.8
RUN mkdir /home/build
COPY . /home/build
WORKDIR /home/build
RUN pip install wheel
RUN pip install ./model
RUN pip install .
EXPOSE 8080
ENTRYPOINT [ "python", "backend/lipread/manage.py", "runserver", "8080" ]