#!/usr/bin/env bash

# kill any servers that may be running in the background 
sudo pkill -f runserver

# stop the apache2 server
sudo systemctl stop apache2

# kill frontend servers if you are deploying any frontend
# sudo pkill -f tailwind
# sudo pkill -f node

cd /var/www/optimizer



# install requirements.txt
pip install -r /var/www/optimizer/requirements.txt
# installing the cryptography for SSL modules
pip install cryptography==38.0.4
# Declaring the environment variables
sudo export ENVIRONMENT=production

# restart apache2 server
sudo systemctl restart apache2
