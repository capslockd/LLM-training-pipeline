FROM python:3.10

#Copy scripts
COPY ./scripts /usr/scripts/ 
RUN chmod a+x /usr/scripts/*.sh

#Install Python Dependencies
COPY ./requirements.txt /usr/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /usr/app/requirements.txt

#Copy Project

WORKDIR /usr/scripts

CMD [ "/usr/scripts/entrypoint.sh" ]