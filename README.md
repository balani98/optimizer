# Media Spend Optimizer 
## Description 
**Optimizer Application :** Optimizer Application is a tool that helps businesses analyze historical data and make informed decisions on how to allocate budget or target across various dimensions based on various inputs . Based on historical data and machine learning model , the optimizer helps to detemine the most effective distribution of their budget or arget across these dimensions.
### Modules in Application 
**Explore :** Explore functionality lets the user upload the data file and select the required metrics like dimensions , investment and target on which optimization need to be performed . It provides visualization and Insights on Investment and Target variables for the selected date range and dimensions . 

**Predict :** Predict functionality lets the user perform the model building on Target for all the dimensions based on Investment.The user can consider seasonality or not in model building process . This tab shows statistics and charts for users to understand how well the model fitting took place for all dimensions.

**Optimize :** Optimize functionality lets the user perform budget optimization to get the maximum possible target for the desired budget. It displays the summary metrics and output of budget optimization process comparing budget and target allocation across various dimensions which took part in optimization.

**Goal Seek**: Goal Seek functionality lets the user perform target optimization to achieve the minimum possible budget for the desired target. It displays the output of target optimization process comparing budget and target allocation across various dimensions which took part in optimization. 
### Requirements 
* Python 3.9 (tested under python 3.9.3)
* Django 4.2 ( tested under djnago 4.2.7)
### Installation in local environment
> Make sure Your Public IP address is added in AWS Security Groups . In case you are connected to your organization VPN. Then organization public IP addresses should be added in security groups.

* Clone the git repository : 
https://github.com/CrossmediaHQ/xm-rb-optimizer.git  
* Go Inside the folder :
`cd xm-rb-optimizer`
* Install all required dependencies : 
`pip install -r requirements.txt`
* Get the necessary files from Dev team ad these files to main directory of your project `/xm-rb-optimizer` :
    + `confg.json:`  It contains necessary configurations related to test and production databases . 
    + `Nabler_Django_Key.pem:` This is private key which helps to setup SSH tunnel with RDS database . 
### Running the project 
* set the variable `ENVIRONMENT` in your local environment :
    + Local Environment: `export ENVIRONMENT=local`
    + Test ENVIRONMENT: `export ENVIORNMENT=test`
* Run the command on your CLI : `python manage.py runserver`

### For Test Server deployment 
[Refer here](./Test_server_deployment.md)

### For Production Server deployment 
[Refer here](./Production_server_deployment.md)


### CI-CD documentation for optimizer 
[Refer here](https://docs.google.com/document/d/1Ug8ibrw-muOtOfmdcCbgOqkFJiU3QGY0U_zL0zYWWBA/edit)
### Contributers
* Mudit Kannodia 
* Deepanshu Balani
