FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN curl -o spongebob.zip https://storage.googleapis.com/writegen/spongebob.zip
RUN curl -o vanilla.zip https://storage.googleapis.com/writegen/vanilla.zip
RUN curl -o south_park_small.zip https://storage.googleapis.com/writegen/south_park_small.zip

WORKDIR /writegen-api

RUN unzip /spongebob.zip -d /writegen-api
RUN rm /spongebob.zip
RUN unzip /vanilla.zip -d /writegen-api/models
RUN rm /vanilla.zip
RUN unzip /south_park_small.zip -d /writegen-api/models
RUN rm /south_park_small.zip

COPY . /writegen-api

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "gpt_api.py"]
