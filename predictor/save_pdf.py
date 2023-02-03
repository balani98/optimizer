import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import random
import json
import os
import time
# customizing runtime configuration stored
# in matplotlib.rcParams
plt.rcParams["figure.figsize"] = [8.00, 6.50]
plt.rcParams["figure.autolayout"] = True
PREDICTOR_UPLOAD_FOLDER = "/var/www/optimizer/Predictor_pdf/"



def plot_curve(multi_chart_data, seasonality, cpm_checked, df_score_final, request):
    filename = PREDICTOR_UPLOAD_FOLDER + "predictor_" + request.session.get("_uuid") + ".pdf"  
    # first remove this file from system if it already exists 
    if os.path.isfile(filename):
        os.remove(filename)
        time.sleep(5)
    keys = multi_chart_data.keys()
    index = 0
    # PdfPages is a wrapper around pdf 
    # file so there is no clash and create
    # files with no error.
    p = PdfPages(filename)
    for key in keys:
        # x axis values
        x = multi_chart_data[key]['spend']
        y2 = multi_chart_data[key]['target']
        # corresponding y axis values
        if seasonality == 1:
            y = multi_chart_data[key]['spend_prediction']
        else :
            y = multi_chart_data[key]['predictions']
        # changing x with cpm
        if cpm_checked == 'True':
            x = multi_chart_data[key]['impression']
        else :
            x = multi_chart_data[key]['spend']
        fig = plt.figure(clear=True)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # plotting the points spend vs predictions
        ax1.plot(x, y)
        # plotting spend vs target 
        ax1.scatter(x, y2, color = 'grey', linewidths = 1,
             s = 10)
        # adding the legend
        ax1.legend(['predictions','data'], loc ="upper left")
        # plotting the grid lines 
        ax1.grid(True)
        # naming the x axis
        if cpm_checked == 'True':
            ax1.set_xlabel('impression')
        else:
            ax1.set_xlabel('spend')
        # naming the y axis
        ax1.set_ylabel('predictions')
        # giving a title to my graph[0]
        #ax = plt.gca()
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
        ax1.set_title(key)
        # saving the plot
        ax2.axis('tight')
        ax2.axis('off')
        df = df_score_final[
            (df_score_final["dimension"] == key)
            ]
        predictor_data_json = json.loads(df.to_json(orient = 'records'))[0]
        table_data = [
            ["consideration","metrics"]
        ]
        for col in predictor_data_json.keys():
            if col == 'SMAPE' or col == 'correlation':
                table_data.append([col, str(int(predictor_data_json[col]*100)) + "%"])
            elif col == '%_of_data_points_discarded_during_outlier_treatment':
                  table_data.append([col, str(predictor_data_json[col]) + "%"])
            else:
                table_data.append([col, str(predictor_data_json[col])])

        the_table = ax2.table(cellText=table_data,loc='center') 
        save_image(p , fig)    
        index = index + 1
   
    p.close()
    return 'curve successfully plotted'

def save_image(p , fig):
    # and saving the files
    fig.savefig(p, format='pdf') 
