FROM bitnami/python:3.8

WORKDIR /app

RUN pip install label-studio

ENV LABEL_STUDIO_BASE_DATA_DIR=/data
ENV LABEL_STUDIO_PASSWORD=label
ENV LABEL_STUDIO_USERNAME=default_user@localhost


EXPOSE 8080

CMD ["label-studio"]