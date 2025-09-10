# #!/bin/sh
# echo "Retrieving certificate"
# export CA_PEM="$(aws ssm get-parameter --name "ea_ca_pem" --region "ap-southeast-2" --with-decryption --query 'Parameter.Value' --output text)"
# if [[ -z "${CA_PEM}" ]]; then
#     echo "No certificate being installed."
# else
#     echo "${CA_PEM}" > /usr/local/share/ca-certificates/eaCA.crt
#     update-ca-certificates
#     export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
# fi
# echo "===> Retrieving ECS Task Metadata"
# export ECS_CONTAINER_TASK_METADATA=$(curl ${ECS_CONTAINER_METADATA_URI}/task)
# echo $ECS_CONTAINER_TASK_METADATA

# echo "===> Proxy Configuration"
# echo $HTTP_PROXY
# echo $HTTPS_PROXY
# echo $NO_PROXY

# echo "===> Launching Query Batch Worker"
# python "$@"
python /usr/scripts/download_dataset.py