FROM python:3.10

# Install CA certificates for SSL
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /usr/scripts

# Copy scripts & requirements
COPY ./scripts /usr/scripts/
RUN chmod +x /usr/scripts/*.sh
COPY ./requirements.txt /usr/scripts/requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r /usr/scripts/requirements.txt

# Default command runs the entrypoint
CMD ["/usr/scripts/entrypoint.sh"]