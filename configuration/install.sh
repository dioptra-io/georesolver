#!/bin/bash

STORAGE_PATH=$1
CONFIG_PATH=$2
EXEC_PATH="$CONFIG_PATH/clickhouse"

if [ $# -ne 2 ]; 
    then 
        echo "Script install.sh requires two input parameters"
        return 1
fi

if [ -z "$STORAGE_PATH" ];
    then 
        echo "Missig first parameter STORAGE_PATH when executing script"
        return 1
fi

if [ -z "$CONFIG_PATH" ];
    then 
        echo "Missig second parameter CONFIG_PATH when executing script"
        return 1
fi

# load env var if not global
source .env

# check if clickhouse credentials were correctly set
if [[ -z "$CLICKHOUSE_DATABASE" || -z "$CLICKHOUSE_USERNAME" || -z "$CLICKHOUSE_PASSWORD" ]]; 
    then
        echo "CLICKHOUSE credentials env var must be set up"
        return 1
fi

if [[ -z "$CLICKHOUSE_ADMIN_USERNAME" || -z "$CLICKHOUSE_ADMIN_PASSWORD" ]]; 
    then
        echo "CLICKHOUSE ADMIN credentials env var must be set up"
        return 1
fi

# check if clickhouse client installed
if [ -f $EXEC_PATH ]; then
    echo "Clickhouse client file already downloaded"
else
    echo "Installing Clickhouse from exec file"
    curl https://clickhouse.com/ | shd
    mv clickhouse $CONFIG_PATH
fi

# start clickhouse server
echo "testing Clickhouse server connectivity"
curl http://0.0.0.0:8123

if [ "$?" -eq 0 ]; then 
    echo "A Clickhouse server is already runnning"
else
    echo "Starting Clickhouse server"

    # pull the docker image
    docker pull clickhouse/clickhouse-server:22.6

    docker run -d \
        -p 8123:8123 \
        -p 8443:8443 \
        -p 127.0.0.1:9000:9000 \
        -p 127.0.0.1:9009:9009 \
        -e CLICKHOUSE_MAX_QUERY_SIZE=10000000000000000000 \
        -v $STORAGE_PATH/data:/var/lib/clickhouse/ \
        -v $STORAGE_PATH/logs:/var/log/clickhouse-server/ \
        -v $CONFIG_PATH/users.d:/etc/clickhouse-server/users.d/ \
        --ulimit nofile=262144:262144 \
        clickhouse/clickhouse-server:22.6

    sleep 2
    
fi

echo "Creating GeoResolver database and user"
# create project database, user and user grants
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE DATABASE IF NOT EXISTS $CLICKHOUSE_DATABASE"
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE USER IF NOT EXISTS $CLICKHOUSE_USERNAME IDENTIFIED WITH plaintext_password BY '$CLICKHOUSE_PASSWORD'"
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT ALL ON $CLICKHOUSE_DATABASE.* TO $CLICKHOUSE_USERNAME WITH GRANT OPTION"
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT ALL ON $CLICKHOUSE_DATABASE_EVALUATION.* TO $CLICKHOUSE_USERNAME WITH GRANT OPTION"

echo "Creating GeoResolver_dev database and user"
# create invited user and db
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE USER IF NOT EXISTS $CLICKHOUSE_USERNAME_DEV IDENTIFIED WITH plaintext_password BY '$CLICKHOUSE_PASSWORD_DEV'"
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE DATABASE IF NOT EXISTS $CLICKHOUSE_DATABASE_DEV"
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT ALL ON $CLICKHOUSE_DATABASE_DEV.* TO $CLICKHOUSE_DEV_USERNAME WITH GRANT OPTION"
    