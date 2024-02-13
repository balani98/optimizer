## Production Server Deployments


### Deployment on production server in case of CI-CD failures
> Makes sure your public IP address  is added in AWS security groups 
* SSH the production server EC2 instance using SSH client . 
* Go to folder /var/www/optimizer/ 
* Copy all the files here from latest commit of main branch .
* Add the same necessary files as mentioned in setting up local environment .
* Restart the apache2 server : 

**Replacement of SSL certificates for production server :** If SSL certificates are expired , Get latest CRT file from Crossmedia IT and replace it with the existing CRT file located at `/etc/apache2/certificates-prod-new/` . Change the path of CRT files at `etc/apache2/sites-enabled/000-default.conf` . Restart the apache2 server : `sudo systemctl restart apache2`. 

**Checking the Logs in production server:** In case of any failures , you can check the apache2 logs at `/var/log/apache2/error.log`

### What to do in case of EC2 failures 
* Try to shut down and start the EC2 again . 
* If it does not work , take the last snapshot backup of the EC2 instance and create an AMI out of that . with the help of AMI , create another EC2 instance and shut down the previous EC2 instance .
* Reassign the Elastic IP address to new instance . 
* Now SSH to the new instance with same details and just restart the apache2 server :`systemctl restart apache2`
* In case above steps do not work , try to follow Fresh deployment of production server .

### Fresh Deployment of Prodution Server
* Spin up an EC2 instance with confguration c4.xlarge and volume of 50 GB .
* Use the following details  : 
    + VPC : Optimizer-VPC-Production
    + Subnet : Nabler_optimizer_production_subnet
    + Security groups : Nabler_Production_security_group
* Use the key pair : Nabler_django_key
* Shutdown the previous EC2 and attach Elastic IP address to new EC2 . 
* When the EC2 is ready , SSH into it . 
* Install the apache2,python and wsgi on ubuntu server using below commands : 
`sudo apt-get update && sudo apt-get install python3-pip apache2 libapache2-mod-wsgi-py3`
* Now go to directory `var/www` , create folder optimizer : `mkdir optimizer` . 
* Do `cd optimizer` 
* Now either copy the code from latest commit of main branch here or clone the code from github here . 
* Install all the requirements : `pip install -r requirements.txt`
* Enable the SSL on EC2 : `sudo a2enmod ssl`
* Enable the WSGI on EC2 : `sudo a2enmod wsgi`
* Install the cryptography module from pip : `python3 -m pip install cryptography==38.0.4`
* Install the MySQL client : `sudo apt-get install libmysqlclient-dev && pip install mysqlclient`
* Copy the CRT and key files  `/etc/apache2/certificates-prod-new`
* Do edits in  `/etc/apache2/sites-enabled/000-default.conf`
* Restart the apache2 server : `sudo systemctl restart apache2`