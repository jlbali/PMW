# Passive Microwave Library of Regridding Methods


## Workflow:

### Requirements and previous installations:

- Python 3.6 or newer.
- python3.6-dev package:

   sudo apt-get install python3.6-dev

- GDAL/OGR

sudo add-apt-repository ppa:ubuntugis/ppa

sudo apt-get update

sudo apt-get install gdal-bin

sudo apt-get install libgdal-dev

export CPLUS_INCLUDE_PATH=/usr/include/gdal

export C_INCLUDE_PATH=/usr/include/gdal



### First Execution:

- Run ./venv_create.sh
- Run source ./venv_activate.sh
- Run ./venv_install.sh
- Run 

pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') 




### Activation of the environment

- Run source ./venv_activate.sh

Prompt should reflect the change by showing "env"

### Scripts

Exp1.py runs the first experiment shown in the paper: "Spatial Correlation Scenario". Exp2.py runst the second experiment: "Observational Error Scenario".
You must configure in those files the desired band and the parameters of the experiment, like the number of samples per image.

