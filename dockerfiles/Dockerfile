FROM --platform=linux/amd64 quay.io/sustainable_computing_io/kepler_model_server_base:v0.7

WORKDIR /usr/local

RUN mkdir -p /usr/local/src
RUN mkdir -p /usr/local/resource

COPY src/estimate src/estimate
COPY src/server src/server
COPY src/train src/train
COPY src/util src/util
COPY cmd cmd

# port for Model Server
ENV GUNICORN_PORT=8105

EXPOSE ${GUNICORN_PORT}
# port for Online Trainer (TODO: reserved for event-based online training)
#EXPOSE 8101
# port for Offline Trainer
#EXPOSE 8102


CMD ["gunicorn", "--config", "config/gunicorn_config.py", "src.server.model_server:app"]