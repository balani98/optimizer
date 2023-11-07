#!/usr/bin/env bash

# kill any servers that may be running in the background 
sudo pkill -f runserver

# stop the apache2 server
sudo systemctl stop apache2

# kill frontend servers if you are deploying any frontend
# sudo pkill -f tailwind
# sudo pkill -f node

cd /mnt/apps/optimizer/source


# install requirements.txt
pip install -r /mnt/apps/optimizer/source/requirements.txt

# installing the cryptography for getting SSL modules
pip install cryptography==38.0.4

# restart apache2 server for production server
# sudo systemctl restart apache2

# restart python server 
screen -dm bash -c  'nohup python manage.py runserver 0.0.0.0:8080' 
