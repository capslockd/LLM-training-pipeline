FROM python:3.10

COPY ./scripts /usr/scripts/
RUN chmod +x /usr/scripts/*.sh

COPY ./requirements.txt /usr/scripts/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /usr/scripts/requirements.txt

WORKDIR /usr/scripts

ENTRYPOINT ["/usr/scripts/entrypoint.sh"]