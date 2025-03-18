#!/bin/bash
CLICKHOUSE_PATH="/storage/clickhouse"
CONFIG_PATH="$PWD/configuration/clickhouse"
EXEC_PATH="$CONFIG_PATH/clickhouse"

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
    curl https://clickhouse.com/ | sh
    mv clickhouse $CONFIG_PATH
fi

# start clickhouse server
echo "testing Clickhouse server connectivity"
curl http://localhost:8123

if [ "$?" -eq 1 ]; then 
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
        -v $CLICKHOUSE_PATH/data:/var/lib/clickhouse/ \
        -v $CLICKHOUSE_PATH/logs:/var/log/clickhouse-server/ \
        -v $CONFIG_PATH/users.d:/etc/clickhouse-server/users.d/ \
        --ulimit nofile=262144:262144 \
        clickhouse/clickhouse-server:22.6

    sleep 2
    
    # create project database, user and user grants
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE DATABASE IF NOT EXISTS $CLICKHOUSE_DATABASE"
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE USER IF NOT EXISTS $CLICKHOUSE_USERNAME IDENTIFIED WITH plaintext_password BY '$CLICKHOUSE_PASSWORD'"
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT ALL ON $CLICKHOUSE_DATABASE.* TO $CLICKHOUSE_USERNAME WITH GRANT OPTION"
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT ALL ON $CLICKHOUSE_DATABASE_EVALUATION.* TO $CLICKHOUSE_USERNAME WITH GRANT OPTION"

    # create invited user
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE USER IF NOT EXISTS $CLICKHOUSE_DEV_USERNAME IDENTIFIED WITH plaintext_password BY '$CLICKHOUSE_DEV_PASSWORD'"
    -- Allow SELECT on specific database or tables
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT SELECT ON $CLICKHOUSE_DATABASE.* TO read_write_create"
    -- Allow INSERT
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT INSERT ON $CLICKHOUSE_DATABASE.* TO read_write_create"
    -- Allow CREATE TABLE
    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT CREATE TABLE ON $CLICKHOUSE_DATABASE.* TO read_write_create"

    $EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "GRANT read_write_create TO $CLICKHOUSE_DEV_USERNAME"
    

fi