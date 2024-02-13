### Deployment on test server in case of CI-CD failures
> Make sure your Public IP address is added in AWS security groups .
* SSH the test server EC2 instance using SSH client.
* Go to folder /mnt/apps/optimizer/source .
* Copy all the files here from latest commit of test branch .
* Add the same necessary files as mentioned in setting up local environment .
* Now check the previous running scripts using command : `htop`
* Kill the previous running scripts : `pkill -f manage.py`
* set the ENVIRONMENT variables : `export ENVIRONMENT=test`
* run the app on PORT 8080: `python manage.py runserver 0.0.0.0:8080`