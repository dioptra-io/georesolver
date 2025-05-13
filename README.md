# GeoResolver

Internet scale IP address geolocation tool. Measurement are run on RIPE Atlas.

## Install project

```bash
cd georesolver
poetry lock && poetry install
```

## Activate virtual env

```bash
poetry shell
```

### Set up .env

GeoResolver relies on .env file to load credentials. 
see env.example for an example of the necessary environment variable. 


### run local script

Once in the poetry virtual env, one can run georesolver using pre-defined scripts:

```bash
python georesolver/scripts/local_demo.py
```

Or execute GeoResolver with the docker image:

```bash
python georesolver/scripts/local_docker_demo.py
```
