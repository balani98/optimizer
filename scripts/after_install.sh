#!/usr/bin/env bash

# kill any servers that may be running in the background 
sudo pkill -f runserver

# stop the apache2 server
sudo systemctl stop apache2

# kill frontend servers if you are deploying any frontend
# sudo pkill -f tailwind
# sudo pkill -f node

cd /var/www/optimizer

# activate virtual environment
python3 -m venv venv
source venv/bin/activate

install requirements.txt
pip install -r /var/www/optimizer/requirements.txt

# Declaring the environment variables
export ENVIRONMENT='production'

# restart apache2 server
sudo systemctl restart apache2
