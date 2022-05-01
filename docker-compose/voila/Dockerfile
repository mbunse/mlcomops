FROM bitnami/python:3.8

RUN pip install voila ipykernel

EXPOSE 8888

VOLUME [ "/data" ]

WORKDIR /data
CMD ["voila", "--port", "8888", "--no-browser"]