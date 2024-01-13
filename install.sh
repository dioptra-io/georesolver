# pull the docker image
docker pull clickhouse/clickhouse-server:22.6


# start the server using docker
docker run -d \
    -v /srv/clickhouse/data:/var/lib/clickhouse/ \
    -p 8123:8123 \
    -p 9000:9000 \
    --ulimit nofile=262144:262144 \
    clickhouse/clickhouse-server:22.6

# download clickhouse client binary
curl https://clickhouse.com/ | sh
mv clickhouse ./clickhouse_files/

# install source files
poetry lock 
poetry install

# run clickhouse db installer for table init
poetry run python scripts/utils/clickhouse_installer.py
