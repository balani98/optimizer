import pandas as pd
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')

#isolate function
def s_curve_hill_inv(Y, a, b, c):
        """This method performs the inverse of scurve function on param, target and
        Returns the outcome as investment"""
        precision = 1e-0
        Y = (Y-(precision/100)) if(Y==c) else Y
        if (Y<=0):
            return 0
        else:
            return ((Y * (b ** a))/(c - Y)) ** (1/a)
    
def dimension_bound(df_param, df_grp, constraint_type):

    """bounds for optimizer

    Returns:
        dictionary: key dimension value [min,max] / [min,max,cpm]
    """

    constraint_type = constraint_type.lower()
    threshold = [0, 3]

    df_param_opt = df_param.T
    df_param_opt.columns = df_param_opt.iloc[0, :]

    d_param = df_param_opt.iloc[1:, :].to_dict()

    dim_bound = {}

    if "cpm" in df_param.columns:

        if constraint_type == "median":
            const_var = "impression_median"
        else:
            const_var = "impression_mean"

        for dim in d_param.keys():
            
            dim_min_imp=int(round(df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['impression'])!=0)]['impression'].min()))
            dim_min_imp_per=-int(round((1-(dim_min_imp/d_param[dim][const_var]))*100))
            dim_bound[dim] = [
                int(dim_min_imp * d_param[dim]["cpm"] / 1000),
                int(
                    (d_param[dim][const_var] * d_param[dim]["cpm"] / 1000)
                    * threshold[1]
                ),
                int(round(d_param[dim][const_var] * d_param[dim]["cpm"] / 1000)),
                dim_min_imp_per,
                200,
                int(d_param[dim]["cpm"])
            ]
    else:
        if constraint_type == "median":
                const_var = "median spend"
        else:
            const_var = "mean spend"

        for dim in d_param.keys():
            
            dim_min_budget=int(round(df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['spend'])!=0)]['spend'].min()))
            dim_min_budget_per=-int(round((1-(dim_min_budget/d_param[dim][const_var]))*100))
            
            dim_bound[dim] = [
                dim_min_budget,
                int(d_param[dim][const_var] * threshold[1]),
                int(round(d_param[dim][const_var])),
                dim_min_budget_per,
                200
            ]

    return dim_bound

def s_curve_hill(X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))

def get_seasonality_conversions(d_param, init_weekday, init_month):

        seasonality_conversion = 0
        for dim in d_param:
            wcoeff = list(d_param[dim].values())[3:9]
            mcoeff = list(d_param[dim].values())[9:20]
            seasonality_conversion_dim = (wcoeff[0] * init_weekday[0]
                                            + wcoeff[1] * init_weekday[1]
                                            + wcoeff[2] * init_weekday[2]
                                            + wcoeff[3] * init_weekday[3]
                                            + wcoeff[4] * init_weekday[4]
                                            + wcoeff[5] * init_weekday[5]
                                            + mcoeff[0] * init_month[0]
                                            + mcoeff[1] * init_month[1]
                                            + mcoeff[2] * init_month[2]
                                            + mcoeff[3] * init_month[3]
                                            + mcoeff[4] * init_month[4]
                                            + mcoeff[5] * init_month[5]
                                            + mcoeff[6] * init_month[6]
                                            + mcoeff[7] * init_month[7]
                                            + mcoeff[8] * init_month[8]
                                            + mcoeff[9] * init_month[9]
                                            + mcoeff[10] * init_month[10])
            seasonality_conversion = seasonality_conversion + seasonality_conversion_dim

        return seasonality_conversion

def conversion_bound(df_param, df_grp, df_bounds, lst_dim, is_seasonality, date_range, is_weekly_selected, convert_to_weekly):

    """bounds for conversion optimizer

    Returns:
        array: max and min number conversions - max is only considered to be 95% of max conversions
    """
   
    if convert_to_weekly ==True:
        is_weekly_selected = True
    print(df_bounds, lst_dim, is_weekly_selected, is_seasonality, convert_to_weekly,date_range)
    if (len(lst_dim)==0):
        return [0, 0]
    df_bounds={dim: df_bounds[dim] for dim in lst_dim}
    
    df_param_=df_param[df_param['dimension'].isin(lst_dim)]
    
    df_param_opt = df_param_.T
    df_param_opt.columns = df_param_opt.iloc[0, :]

    d_param = df_param_opt.iloc[1:, :].to_dict()

    dim_conversion = {}
    min_conversion = 0
    max_conversion = 0
    
    for dim in d_param.keys():
        if "cpm" in df_param.columns:
            
            dim_min_inp_budget=int(df_bounds[dim][0])
            dim_max_inp_budget=int(df_bounds[dim][1])
            dim_min_budget=dim_min_inp_budget

            dim_min_imp=(dim_min_budget * 1000) / df_bounds[dim][2]
            dim_min_conversion=s_curve_hill(dim_min_imp, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            
            # Getting maximum conversions
            dim_max_poss_conversion=int(d_param[dim]["param c"])
            dim_max_inp_imp = (dim_max_inp_budget * 1000) / df_bounds[dim][2]
            dim_max_inp_conversion=s_curve_hill(dim_max_inp_imp, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            if (dim_max_inp_conversion>=dim_max_poss_conversion):
                dim_max_conversion=dim_max_poss_conversion
            else:
                dim_max_conversion=dim_max_inp_conversion
            
        else:        
            dim_min_inp_budget=int(df_bounds[dim][0])
            dim_max_inp_budget=int(df_bounds[dim][1])
            dim_min_budget=dim_min_inp_budget

            dim_min_conversion=s_curve_hill(dim_min_budget, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            
            # Getting maximum conversions
            dim_max_poss_conversion=int(d_param[dim]["param c"])
            dim_max_inp_conversion=s_curve_hill(dim_max_inp_budget, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            if (dim_max_inp_conversion>=dim_max_poss_conversion):
                dim_max_conversion=dim_max_poss_conversion
            else:
                dim_max_conversion=dim_max_inp_conversion
                
        dim_conversion[dim]=[dim_min_conversion, dim_max_conversion]
        min_conversion=min_conversion+dim_conversion[dim][0]
        max_conversion=max_conversion+dim_conversion[dim][1]

    if is_seasonality == True:
        days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1
        if is_weekly_selected == True:
            days = int(days/7)
            day_name = pd.to_datetime(date_range[0]).day_name()[0:3]
            freq_type = "W-"+day_name
        else:
            freq_type = "D"
        seasonality_combination = []
        seasonality_count = {}
        for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both", freq=freq_type):
            day_counter = str(day_.weekday())+"_"+str(day_.month)
            seasonality_combination = seasonality_combination + [day_counter]
            if day_counter in seasonality_count.keys():
                seasonality_count[day_counter] += 1
            else:
                seasonality_count[day_counter] = 1
        seasonality_combination = set(seasonality_combination)

        overall_seas_conv = 0
        for day_month in seasonality_combination:

            init_weekday = [0, 0, 0, 0, 0, 0]
            init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            weekday = int(day_month.split('_')[0])
            month = int(day_month.split('_')[1])
                
            if weekday != 0:
                init_weekday[weekday - 1] = 1

            if month != 1:
                init_month[month - 2] = 1

            overall_seas_conv = (overall_seas_conv
                                + (get_seasonality_conversions(d_param, init_weekday, init_month)
                                    * seasonality_count[day_month]))
            
        return [int(np.ceil((min_conversion*days) + overall_seas_conv)), int(np.floor((max_conversion*days) + overall_seas_conv))]
    else:
        days=date_range 
        return [int(np.ceil(min_conversion)*days), int(np.floor(max_conversion)*days)]

class optimizer_conversion:
    def __init__(self, df_param, constraint_type, target_type):
        """initialization

        Args:
            df_param (dataframe): model param
        """
        df_param_opt = df_param.T
        df_param_opt.columns = df_param_opt.iloc[0, :]

        self.d_param = df_param_opt.iloc[1:, :].to_dict()

        self.constraint_type = constraint_type.lower()
        self.target_type = target_type.lower()

        if "cpm" in df_param.columns:
            self.use_impression = True
            self.metric = 'impression'
            if self.constraint_type == 'median':
                self.const_var = 'impression_median'
            else:
                self.const_var = 'impression_mean'
        else:
            self.use_impression = False
            self.metric = 'spend'
            if self.constraint_type == 'median':
                self.const_var = 'median spend'
            else:
                self.const_var = 'mean spend'        
            
        self.dimension_names = list(self.d_param.keys())
        # Precision used for optimization
        self.precision = 1e-0
        # Max iterations used for optimization
        self.max_iter = 50000

    def s_curve_hill(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))
    
    def s_curve_hill_inv(self, Y, a, b, c):
        """This method performs the inverse of scurve function on param, target and
        Returns the outcome as investment"""
        Y = (Y-(self.precision/100)) if(Y==c) else Y
        if (Y<=0):
            return 0
        else:
            return ((Y * (b ** a))/(c - Y)) ** (1/a)
    
    def ini_start_value(self, df_grp, dimension_bound):
        """initialization of initial metric (spend or impression) to overcome the local minima for each dimension
        
        Returns:
            Array - float value:
                For impression: Minimum impression and corresponding spend for each dimension
                For spend: Minimum spend for each dimension
        """
        oldSpendVec = {}
        oldImpVec = {}
        
        for dim in self.dimension_names:
            if self.use_impression:
                df_grp_tmp_imp = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['impression'])!=0)].copy()
                start_value_imp = df_grp_tmp_imp[self.metric].min()
                start_value_spend=(start_value_imp*dimension_bound[dim][2])/1000
            
                input_start_imp=((dimension_bound[dim][0] * 1000) / dimension_bound[dim][2])
                input_start_spend=dimension_bound[dim][0]
                
                input_final_imp=((dimension_bound[dim][1] * 1000) / dimension_bound[dim][2])
                input_final_spend=dimension_bound[dim][1]
                
                if(input_start_spend>start_value_spend):
                    oldImpVec[dim]=input_start_imp
                    oldSpendVec[dim]=input_start_spend
                elif(input_final_spend<start_value_spend):
                    oldImpVec[dim]=input_start_imp
                    oldSpendVec[dim]=input_start_spend
                else:
                    oldImpVec[dim]=start_value_imp
                    oldSpendVec[dim]=start_value_spend
            else:
                df_grp_tmp_spend = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['spend'])!=0)].copy()
                start_value_spend = df_grp_tmp_spend['spend'].min()
                
                input_start_spend=dimension_bound[dim][0]
                input_final_spend=dimension_bound[dim][1]
            
                if(input_start_spend>start_value_spend):
                    oldSpendVec[dim]=input_start_spend
                elif(input_final_spend<start_value_spend):
                    oldSpendVec[dim]=input_start_spend
                else:
                    oldSpendVec[dim]=start_value_spend
                
        if self.use_impression:
            return oldSpendVec, oldImpVec
        else:
            return oldSpendVec
        

    def increment_factor(self, df_grp):
        """Increment value for each iteration
        
        Returns:
            Float value: Increment factor - always based on spend (irrespective of metric chosen)
        """
        # inc_factor =  df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby('date').agg({'spend':'sum','target':'sum'})['spend'].median()
        # increment = round(inc_budget*0.075)
        # increment = round(df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby(['dimension']).agg({'spend':'median'})['spend'].median())
        inc_factor = round(df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby(['dimension']).agg({'spend':self.constraint_type})['spend'].min())
        increment = round(inc_factor*0.50)
        return increment
    
    
    def initial_conversion(self, oldMetricVec):
        """initialization of initial conversions for each dimension for initail slected metric (spend or impression)
        
        Returns:
            Array - float value:
                Conversions
        """
        oldReturn = {}
        for dim in self.dimension_names:
            oldReturn[dim]=(self.s_curve_hill(oldMetricVec[dim],
                                          self.d_param[dim]["param a"],
                                          self.d_param[dim]["param b"],
                                          self.d_param[dim]["param c"]))
        return oldReturn
    

    def get_conversion_dimension(self, newSpendVec, dimension_bound, increment, newImpVec):
        """Function to get dimension and their conversion for increment budget - to derive dimension having maximum conversion
        
        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension
                totalReturn: Conversion for allocated budget for each dimension          
        """
        incReturns = {}
        incBudget = {}
            
        for dim in self.dimension_names:

            oldSpendTemp = newSpendVec[dim]
            if self.use_impression:
                oldImpTemp = newImpVec[dim]  
                        
            # check if spend allocated to a dimension + increment is less or equal to max constarint and get incremental converstions
            if((newSpendVec[dim] + increment)<=dimension_bound[dim][1]):
                incBudget[dim] = increment
            # check if spend allocated to a dimension + increment is greater than max constarint and get converstions for remaining budget for that dimension
            elif((newSpendVec[dim]<dimension_bound[dim][1]) & ((newSpendVec[dim] + increment)>dimension_bound[dim][1])):
                # getting remaining increment budget if post increment allocation budget exceeds max bound for a dimension
                incBudget[dim] = dimension_bound[dim][1] - newSpendVec[dim]
            # if max budget is exhausted for that dimension
            else:
                incBudget[dim]=0
                incReturns[dim]=-1
                continue
        
            # updated spend post increment budget allocation
            newSpendTemp = newSpendVec[dim] + incBudget[dim]

            # check for increment return
            if self.use_impression:
                newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                incReturns[dim]=(self.s_curve_hill(newImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                            -  self.s_curve_hill(oldImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            else:
                incReturns[dim]=(self.s_curve_hill(newSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                            -  self.s_curve_hill(oldSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))  

        return incReturns, incBudget
    
    
    def conversion_optimize_spend(self, increment, oldSpendVec, oldReturn, returnGoal, dimension_bound):
        """function for calculating target conversion when metric as spend is selected
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend and Conversion for every dimension
                Iterration Result: Number of iterations and corresponding spend and conversion to reach target conversion (result not used in the UI)
            Message 4001: Optimum conversion reached
            Message 4002: Exceeded number of iterations, solution couldn't be found
        """
        results_itr_df=pd.DataFrame(columns=['spend', 'return'])
        
        newSpendVec = oldSpendVec.copy()
        totalSpend = sum(oldSpendVec.values())
        totalReturn = oldReturn.copy()

        if self.use_impression:
            newImpVec = oldImpVec.copy()
        else:
            newImpVec = {}
        
        result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
        results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
        results_itr_df=results_itr_df.reset_index(drop=True)
        
        iteration=0
        itr_calibrate=0
        calibrate_flag=0
        msg=4001
        
        while(returnGoal>sum(totalReturn.values())):                   
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, newImpVec)    
            dim_idx=max(incReturns, key=incReturns.get)
            
            if(incReturns[dim_idx]>0):
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                totalReturn[dim_idx] = self.s_curve_hill(newSpendVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                returnGoal_err_per = (sum(totalReturn.values()) - returnGoal) * 100 / returnGoal
                # print("Error rate ",sum(totalReturn.values())," ",returnGoal," ",returnGoal_err_per)
                
                if((returnGoal_err_per>=-0.5) and (returnGoal_err_per<=1.5)):
                    msg=4001
                    iteration+=1
                    # print("inc - cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    break
                    
                elif((returnGoal_err_per>1.5) and (itr_calibrate<=500)):
                    itr_calibrate+=1
                    # print("not inc - cal - greater 1.5%"," ",dim_idx," ",increment)
                    newSpendVec[dim_idx] = newSpendVec[dim_idx] - incBudget[dim_idx]
                    totalReturn[dim_idx] = self.s_curve_hill(newSpendVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    calibrate_increment=round(increment*0.1)
                    increment=increment-calibrate_increment
                    
                elif(itr_calibrate>500):
                    msg=4003
                    print("not inc - cal - cal not possible"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    raise Exception("Optimal solution not found")
                    break
                
                else:
                    msg=4001
                    iteration+=1
                    # print("inc - not cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                
            elif(incReturns[dim_idx]==-1):
                iteration+=1
                itr_calibrate+=1
                # print("not inc - increment calibartion"," ",increment)
                calibrate_increment=round(increment*0.1)
                increment=increment-calibrate_increment
            
            if(iteration>=self.max_iter):
                msg=4002
                raise Exception("Optimal solution not found")
                break
        
        newSpendVec, totalReturn, newImpVec = self.adjust_budget(newSpendVec, totalReturn, dimension_bound, None)
        # print(iteration," ",itr_calibrate)
        conversion_return_df = pd.DataFrame(totalReturn.items())
        budget_return_df = pd.DataFrame(newSpendVec.items())
        
        conversion_return_df.rename({0: 'dimension', 1: 'return'}, axis=1, inplace=True)
        budget_return_df.rename({0: 'dimension', 1: 'spend'}, axis=1, inplace=True)
        result_df = pd.merge(budget_return_df, conversion_return_df, on='dimension', how='outer')

        return result_df, results_itr_df, msg
    
    
    def conversion_optimize_impression(self, increment, oldSpendVec, oldImpVec, oldReturn, returnGoal, dimension_bound):
        """function for calculating target conversion when metric as impression is selected
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend, Impression and Conversion for every dimension
                Iterration Result: Number of iterations and corresponding spend, impression and conversion to reach target conversion (result not used in the UI)
            Message 4001: Optimum conversion reached
            Message 4002: Exceeded number of iterations, solution couldn't be found
        """
        results_itr_df=pd.DataFrame(columns=['spend', 'impression', 'return'])
        
        newSpendVec = oldSpendVec.copy()
        newImpVec = oldImpVec.copy()
        
        totalSpend = sum(oldSpendVec.values())
        totalImp = sum(oldImpVec.values())
        totalReturn = oldReturn.copy()

        if self.use_impression:
            newImpVec = oldImpVec.copy()
        else:
            newImpVec = {}
        
        result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
        results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
        results_itr_df=results_itr_df.reset_index(drop=True)
        
        iteration=0
        itr_calibrate=0
        msg=4001
        while(returnGoal>sum(totalReturn.values())):
            
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, newImpVec)    
            dim_idx=max(incReturns, key=incReturns.get)
            
            if(incReturns[dim_idx]>0):
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                newImpVec[dim_idx] = ((newSpendVec[dim_idx]*1000)/(dimension_bound[dim_idx][2]))
                totalReturn[dim_idx] = self.s_curve_hill(newImpVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                
                returnGoal_err_per = (sum(totalReturn.values()) - returnGoal) * 100 / returnGoal
                # print(sum(totalReturn.values())," ",returnGoal," ",returnGoal_err_per)
                # print("Error rate ",sum(totalReturn.values())," ",returnGoal," ",returnGoal_err_per)
                
                if((returnGoal_err_per>=-0.5) and (returnGoal_err_per<=1.5)):
                    msg=4001
                    iteration+=1
                    # print("inc - cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    break
                    
                elif((returnGoal_err_per>1.5) and (itr_calibrate<=500)):
                    itr_calibrate+=1
                    # print("not inc - cal - greater 1.5%"," ",dim_idx," ",increment)
                    newSpendVec[dim_idx] = newSpendVec[dim_idx] - incBudget[dim_idx]
                    newImpVec[dim_idx] = ((newSpendVec[dim_idx]*1000)/(dimension_bound[dim_idx][2]))
                    totalReturn[dim_idx] = self.s_curve_hill(newImpVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    calibrate_increment=round(increment*0.1)
                    increment=increment-calibrate_increment
                    
                elif(itr_calibrate>500):
                    msg=4003
                    # print("not inc - cal - cal not possible"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    raise Exception("Optimal solution not found")
                    break
                
                else:
                    msg=4001
                    iteration+=1
                    # print("inc - not cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
            
            elif(incReturns[dim_idx]==-1):
                iteration+=1
                itr_calibrate+=1
                # print("not inc - increment calibartion"," ",increment)
                calibrate_increment=round(increment*0.1)
                increment=increment-calibrate_increment
            
            if(iteration>=self.max_iter):
                msg=4002
                raise Exception("Optimal solution not found")
                break
        
        newSpendVec, totalReturn, newImpVec = self.adjust_budget(newSpendVec, totalReturn, dimension_bound, newImpVec)
        # print(iteration," ",itr_calibrate)
        conversion_return_df = pd.DataFrame(totalReturn.items())
        budget_return_df = pd.DataFrame(newSpendVec.items())
        imp_return_df = pd.DataFrame(newImpVec.items())
        
        conversion_return_df.rename({0: 'dimension', 1: 'return'}, axis=1, inplace=True)
        budget_return_df.rename({0: 'dimension', 1: 'spend'}, axis=1, inplace=True)
        imp_return_df.rename({0: 'dimension', 1: 'impression'}, axis=1, inplace=True)
        
        result_df = pd.merge(imp_return_df, conversion_return_df, on='dimension', how='outer')
        result_df = pd.merge(result_df, budget_return_df, on='dimension', how='outer')

        return result_df, results_itr_df, msg
    

    def total_return(self, newSpendVec, totalReturn, dimension_bound, dim, newImpVec):
        """calculate total spend based on spend or impression
        
        Returns:
            Dictionay - 
                totalReturn: Conversion for allocated budget or impression for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        if self.use_impression:
            newImpVec[dim] = ((newSpendVec[dim]*1000)/(dimension_bound[dim][2]))
            totalReturn[dim] = self.s_curve_hill(newImpVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
        else:
            totalReturn[dim] = self.s_curve_hill(newSpendVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
        return totalReturn, newImpVec


    def adjust_budget(self, newSpendVec, totalReturn, dimension_bound_actual, newImpVec):
        """Budget for each dimension is checked and adjusted based on the following:
            Budget adjust due to rounding error in target
            If a particular dimension has zero target but some budget allocated by optimizer, this scenario occurs when inilization is done before optimization process

        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension after adjustment
                totalReturn: Conversion for allocated budget for each dimension after adjustment
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        """
        Note: No requirement for checking grouped constraints for this function
            Rounding error adjustment: Budget will remain same or floor level budget will be used
            Zero conversion dimension: Budget will be reduced to 0 or lower bound where no conversion is generated
        """
        budgetDecrement = 0

        # adjust budget due to rounding error
        for dim in self.d_param:
            dim_spend=newSpendVec[dim]
            dim_return = 0
            conv=totalReturn[dim]
            if (round(conv)>conv):
                dim_return=np.trunc(conv*10)/10
            elif (round(conv)<conv):
                dim_return=int(conv)
            else:
                continue
            dim_metric = self.s_curve_hill_inv(totalReturn[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
            
            if self.use_impression:
                dim_metric_spend=(newImpVec[dim] * dimension_bound_actual[dim][2])/1000
                if(dim_metric_spend>=dimension_bound_actual[dim][0]):
                    newImpVec[dim] = dim_metric
                    newSpendVec[dim] = dim_metric_spend
                    totalReturn[dim] = dim_return
                else:
                    continue   
            else:
                if(dim_metric>=dimension_bound_actual[dim][0]):
                    newSpendVec[dim] = dim_metric
                    totalReturn[dim] = dim_return
                else:
                    continue
            
            budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)

        # decrement unused budget from dimensions having almost zero conversion as part of budget allocation during initialization of initial budget value
        for dim in self.d_param:
            if ((totalReturn[dim]<1) and (newSpendVec[dim]>0)):
                budgetDecrement = budgetDecrement + (newSpendVec[dim] - dimension_bound_actual[dim][0])
                newSpendVec[dim] = dimension_bound_actual[dim][0]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)

        return newSpendVec, totalReturn, newImpVec
    

    def lift_cal(self, result_df, conv_goal, df_spend_dis, days, dimension_bound):
        """function for calculating different columns to be displayed for n days selected by the user
        
        Returns:
            Dataframe:
                Final Result Dataframe: Contains percentage and n days calculation
        """
        if self.use_impression:
            result_df.rename({'impression': 'recommended_impressions_per_day', 'spend': 'recommended_budget_per_day', 'return': 'estimated_return_per_day'}, axis=1, inplace=True)
            cpm_df=pd.DataFrame([(dim, dimension_bound[dim][2]) for dim in dimension_bound], columns=['dimension', 'cpm'])
            result_df=pd.merge(result_df, cpm_df, on='dimension', how='left')
        else:
            result_df.rename({self.metric: 'recommended_budget_per_day', 'return': 'estimated_return_per_day'}, axis=1, inplace=True)
        
        result_df['recommended_budget_per_day']=result_df['recommended_budget_per_day'].round().astype(int)
        result_df['estimated_return_per_day']=result_df['estimated_return_per_day'].round().astype(int)
        result_df['recommended_budget_for_n_days'] = (result_df['recommended_budget_per_day']*days).round().astype(int)
        result_df['estimated_return_for_n_days'] = (result_df['estimated_return_per_day']*days).round().astype(int)
        result_df['buget_allocation_new_%'] = (result_df['recommended_budget_per_day']/sum(result_df['recommended_budget_per_day'])).round(1)
        result_df['estimated_return_%'] = ((result_df['estimated_return_per_day']/sum(result_df['estimated_return_per_day']))*100).round(1)
                
        result_df=result_df[['dimension', 'recommended_budget_per_day', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days']]
        
        return result_df


    def optimizer_result_adjust(self, discard_json, df_res, df_spend_dis, dimension_bound, conv_goal, days):
        """re-calculation of result based on discarded dimension budget

        Args:
            discard_json (json): key: discarded dimension ,value: spend
            df_res (dataframe): res dataframe from optimizer
            df_spend_dis (dataframe): spend distribution
            days (int): days

        Returns:
            dataframe: recal of optimizer result for discarded dimension
        """
        discard_json = {chnl:discard_json[chnl] for chnl in discard_json.keys() if(discard_json[chnl]!=0)}
        check_discard_json = bool(discard_json)
        d_dis = df_spend_dis.set_index('dimension').to_dict()['spend']

        for dim_ in discard_json.keys():
            l_append = []
            for col in df_res.columns:
                
                l_col_update = ['dimension','recommended_budget_per_day','recommended_budget_for_n_days']

                if(col in l_col_update):
                    if(col=='recommended_budget_per_day'):
                        l_append.append(int(round(discard_json[dim_])))
                    elif(col=='recommended_budget_for_n_days'):
                        l_append.append(int(round(discard_json[dim_]))*days)
                    else:
                        l_append.append(dim_)
                else:
                    l_append.append(None)

            df_res.loc[-1] = l_append
            df_res.index = df_res.index + 1  # shifting index
            df_res = df_res.sort_index()

        df_res['buget_allocation_new_%'] = ((df_res['recommended_budget_per_day']/sum(df_res['recommended_budget_per_day']))*100).round(1)

        df_res = df_res.merge(df_spend_dis[['dimension', 'median spend', 'mean spend', 'spend']], on='dimension', how='left')
        df_res['total_buget_allocation_old_%'] = ((df_res['spend']/df_res['spend'].sum())*100).round(1)

        if self.constraint_type == 'median':
            df_res['buget_allocation_old_%'] = ((df_res['median spend']/df_res['median spend'].sum())*100)
            df_res['median spend'] = df_res['median spend'].round().astype(int)
            df_res = df_res.rename(columns={"median spend": "original_constraint_budget_per_day"})
        else:
            df_res['buget_allocation_old_%'] = ((df_res['mean spend']/df_res['mean spend'].sum())*100)
            df_res['mean spend'] = df_res['mean spend'].round().astype(int)
            df_res = df_res.rename(columns={"mean spend": "original_constraint_budget_per_day"})

        for dim in self.d_param:
            budget_per_day = int(sum(df_res['recommended_budget_per_day']))
            spend_projections = budget_per_day*(df_res.loc[df_res['dimension']==dim, 'buget_allocation_old_%']/100)
            df_res.loc[df_res['dimension']==dim, 'spend_projection_constraint_for_n_day'] = spend_projections * days
            if self.use_impression:
                imp_projections = (spend_projections * 1000)/dimension_bound[dim][2]
                metric_projections = imp_projections
            else:
                metric_projections = spend_projections
            df_res.loc[df_res['dimension']==dim, 'current_projections_per_day'] = self.s_curve_hill(metric_projections, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]).round(2)
        df_res['current_projections_for_n_days'] = df_res['current_projections_per_day']*days
        df_res['current_projections_%'] = ((df_res['current_projections_per_day']/df_res['current_projections_per_day'].sum())*100).round(1)
        df_res['buget_allocation_old_%']=df_res['buget_allocation_old_%'].round(1)
        df_res["spend_projection_constraint_for_n_day"]=df_res["spend_projection_constraint_for_n_day"].round()
        df_res['current_projections_for_n_days']=df_res['current_projections_for_n_days'].round()

        summary_metrics_dic = {"Optimized Total Budget" : sum(df_res['recommended_budget_for_n_days'].replace({np.nan: 0.0}).astype(float).round().astype(int)),
                               "Optimized Total Target" : sum(df_res['estimated_return_for_n_days'].replace({np.nan: 0.0}).astype(float).round().astype(int)),
                               "Current Projection Total Budget" : sum(df_res["spend_projection_constraint_for_n_day"].replace({np.nan: 0.0}).astype(float).round().astype(int)),
                               "Current Projection Total Target" : sum(df_res['current_projections_for_n_days'].astype(float).replace({np.nan: 0.0}).round().astype(int)) 
                               }

        if self.target_type == "revenue":
            summary_metrics_dic["Optimized CPA/ROI"] = round(summary_metrics_dic["Optimized Total Target"]/summary_metrics_dic["Optimized Total Budget"], 2)
            summary_metrics_dic["Current Projection CPA/ROI"] = round(summary_metrics_dic["Current Projection Total Target"]/summary_metrics_dic["Current Projection Total Budget"], 2)
            df_res["current_projections_CPA_ROI"] = ((df_res["current_projections_for_n_days"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["spend_projection_constraint_for_n_day"].replace({np.nan: 0, None: 0})!=0, "spend_projection_constraint_for_n_day"])).replace({np.inf: 0}).round(2)
            df_res["optimized_CPA_ROI"] = ((df_res["estimated_return_per_day"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["recommended_budget_per_day"].replace({np.nan: 0, None: 0})!=0, "recommended_budget_per_day"])).replace({np.inf: 0}).round(2)
        else:
            summary_metrics_dic["Optimized CPA/ROI"] = round(summary_metrics_dic["Optimized Total Budget"]/summary_metrics_dic["Optimized Total Target"], 2)
            summary_metrics_dic["Current Projection CPA/ROI"] = round(summary_metrics_dic["Current Projection Total Budget"]/summary_metrics_dic["Current Projection Total Target"], 2)
            df_res["current_projections_CPA_ROI"] = ((df_res["spend_projection_constraint_for_n_day"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["current_projections_for_n_days"].replace({np.nan: 0, None: 0})!=0, "current_projections_for_n_days"])).replace({np.inf: 0}).round(2)
            df_res["optimized_CPA_ROI"] = ((df_res["recommended_budget_per_day"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["estimated_return_per_day"].replace({np.nan: 0, None: 0})!=0, "estimated_return_per_day"])).replace({np.inf: 0}).round(2)

        df_res["current_projections_CPA_ROI"] = np.where((df_res["spend_projection_constraint_for_n_day"]==0) | (df_res["current_projections_for_n_days"]==0), 0, df_res["current_projections_CPA_ROI"])
        df_res["optimized_CPA_ROI"] = np.where((df_res["recommended_budget_per_day"]==0) | (df_res["estimated_return_per_day"]==0), 0, df_res["optimized_CPA_ROI"])

        if self.constraint_type == 'median':
            df_res = df_res.rename(columns={"original_constraint_budget_per_day": "original_median_budget_per_day"})
            df_res = df_res.rename(columns={"buget_allocation_old_%": "median_buget_allocation_old_%"})
            df_res=df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'median_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%']]
        else:
            df_res = df_res.rename(columns={"original_constraint_budget_per_day": "original_mean_budget_per_day"})
            df_res = df_res.rename(columns={"buget_allocation_old_%": "mean_buget_allocation_old_%"})
            df_res=df_res[['dimension', 'original_mean_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'mean_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%']]
        
        df_res = df_res.replace({np.nan: None})

        for dim in discard_json:
            df_res.loc[df_res['dimension']==dim, 'current_projections_CPA_ROI'] = None
            df_res.loc[df_res['dimension']==dim, 'optimized_CPA_ROI'] = None

        int_cols = [i for i in df_res.columns if ((i != "dimension") & ('%' not in i) & ('CPA_ROI' not in i))]
        for i in int_cols:
            df_res.loc[df_res[i].values != None, i]=df_res.loc[df_res[i].values != None, i].astype(float).round().astype(int)

        return df_res, summary_metrics_dic, check_discard_json
    
        
    def dimension_bound_max_check(self, dimension_bound):
        for dim in dimension_bound:
            # dim_max_poss_conversion=np.floor(self.d_param[dim]["param c"]*0.95)
            dim_max_poss_conversion=int(self.d_param[dim]["param c"])
            
            dim_max_inp_budget=dimension_bound[dim][1]

            if self.use_impression:
                dim_max_inp_imp=(dim_max_inp_budget * 1000) / dimension_bound[dim][2]
                dim_max_inp_conversion=s_curve_hill(dim_max_inp_imp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_imp=int(s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                dim_max_poss_budget=(dim_max_poss_imp * dimension_bound[dim][2])/1000
            else:
                dim_max_inp_conversion=s_curve_hill(dim_max_inp_budget, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_budget=int(s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            
            if (dim_max_inp_conversion>dim_max_poss_conversion):
               dim_max_budget=dim_max_poss_budget
            elif (dim_max_inp_conversion==dim_max_poss_conversion):
               dim_max_budget=min(dim_max_poss_budget, dim_max_inp_budget)
            else:
                dim_max_budget=dim_max_inp_budget

            if(dim_max_budget<dimension_bound[dim][0]):
                dim_max_budget=dimension_bound[dim][0]
            
            dimension_bound[dim][1]=dim_max_budget
        return dimension_bound
    

    def confidence_score(self, result_df, accuracy_df, df_grp, lst_dim, dimension_bound):
        """function for calculating optimization confidence score
        calculation is based on independent accuracy of each dimension (calculated using their SMAPE), % budget allocation over historic max and budget distribution across dimension based on optimization
        % budget allocation over historic max is utlized for penalizing dimensions whose recommended budget is beyond their historic maximum value (with weightage 20%)
        
        Returns:
            Value: Optimization confidence score
        """
        penalty_weightage = 0.20
        # Dimension level accuracy calculation using SMAPE (error in the model)
        accuracy_df['Accuracy'] = 1 - accuracy_df['SMAPE']
        # Maximum historic spend at dimension level
        if self.use_impression:
            max_dim_df = df_grp.groupby('dimension').agg(max_impression=('impression', 'max')).round().reset_index()
            cpm_df=pd.DataFrame([(dim, dimension_bound[dim][2]) for dim in dimension_bound], columns=['dimension', 'cpm'])
            max_dim_df = pd.merge(cpm_df, max_dim_df, on='dimension', how='left')
            max_dim_df['max_spend'] = (max_dim_df['max_impression']*max_dim_df['cpm'])/1000
            max_dim_df = max_dim_df[['dimension', 'max_spend']]
        else:
            max_dim_df = df_grp.groupby('dimension').agg(max_spend=('spend', 'max')).round().reset_index()
        score_df = pd.merge(accuracy_df[['dimension', 'Accuracy']], max_dim_df, on='dimension')
        score_df = pd.merge(score_df, result_df[['dimension', 'recommended_budget_per_day', 'buget_allocation_new_%']], on='dimension')
        # Filtering for participating dimensions in optimization (dimensions selected by the user in frontend)
        score_df = score_df[score_df['dimension'].isin(lst_dim)]
        score_df['buget_allocation_new_%'] = score_df['buget_allocation_new_%']/100

        # Calculating % budget allocation by optimization over maximum historic spend for penalising such dimensions
        score_df['budget_allocated_over_max_%'] = np.where(score_df['recommended_budget_per_day']>score_df['max_spend'],
                                                           ((score_df['recommended_budget_per_day']-score_df['max_spend'])/score_df['max_spend']), 0)
        
        # Adjusting accuracy based on above step
        # Formula used: ((Accuracy x 100%) - (% Budget Allocation over Max Spend x Penalization Weightage)) x % Recommended Budget Allocation
        score_df['score'] = (score_df['Accuracy'] - (score_df['budget_allocated_over_max_%']*penalty_weightage)) * score_df['buget_allocation_new_%']

        # Final optimization confidence score based on weighted average
        conf_score = round((sum(score_df['score'])/sum(score_df['buget_allocation_new_%']))*100, 1)

        return conf_score

                
    def execute(self, df_grp, conv_goal, days, df_spend_dis, discard_json, dimension_bound, lst_dim, df_score_final):
        """main function for calculating target conversion
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend/Impression and Conversion for every dimension
        """
        d_param_old=copy.deepcopy(self.d_param)
        df_param_temp= pd.DataFrame(self.d_param).T.reset_index(drop=False).rename({'index':'dimension'}, axis=1)
        df_param_temp=df_param_temp[df_param_temp['dimension'].isin(lst_dim)].reset_index(drop=True)
        df_param_opt = df_param_temp.T
        df_param_opt.columns = df_param_opt.iloc[0, :]
        d_param = df_param_opt.iloc[1:, :].to_dict()
        self.d_param=d_param

        dimension_bound_old=copy.deepcopy(dimension_bound)
        dim_bnd_temp= pd.DataFrame(dimension_bound).T.reset_index(drop=False).rename({'index':'dimension'}, axis=1)
        dim_bnd_temp=dim_bnd_temp[dim_bnd_temp['dimension'].isin(lst_dim)].reset_index(drop=True)
        dim_bnd_opt = dim_bnd_temp.T
        dim_bnd_opt.columns = dim_bnd_opt.iloc[0, :]
        dimension_bound = dim_bnd_opt.iloc[1:, :].to_dict()

        self.dimension_names = list(self.d_param.keys())        

        dimension_bound = self.dimension_bound_max_check(dimension_bound)
        
        returnGoal = np.round((conv_goal/days),2)
                     
        increment = self.increment_factor(df_grp)
        # increment = round(inc_budget*0.075)
        if self.use_impression:
            oldSpendVec, oldImpVec = self.ini_start_value(df_grp, dimension_bound)
            oldReturn = self.initial_conversion(oldImpVec)
            result_df, result_itr_df, msg = self.conversion_optimize_impression(increment, oldSpendVec, oldImpVec, oldReturn, returnGoal, dimension_bound)
            result_df=result_df[['dimension', 'spend', 'impression', 'return']]
            result_df[['spend', 'impression', 'return']]=result_df[['spend', 'impression', 'return']].round(2)
        else:
            oldSpendVec = self.ini_start_value(df_grp, dimension_bound)
            oldReturn = self.initial_conversion(oldSpendVec)
            result_df, result_itr_df, msg = self.conversion_optimize_spend(increment, oldSpendVec, oldReturn, returnGoal, dimension_bound)
            result_df=result_df[['dimension', 'spend', 'return']]
            result_df[['spend', 'return']]=result_df[['spend', 'return']].round(2)
        
        result_df = self.lift_cal(result_df, conv_goal, df_spend_dis, days, dimension_bound)
        result_df, summary_metrics_dic, check_discard_json = self.optimizer_result_adjust(discard_json, result_df, df_spend_dis, dimension_bound, conv_goal, days)        
        result_itr_df=result_itr_df.round(2)

        # Optimization confidence score calculation
        optimization_conf_score = self.confidence_score(result_df, df_score_final, df_grp, lst_dim, dimension_bound)

        return result_df, summary_metrics_dic, optimization_conf_score, check_discard_json

class optimizer_conversion_seasonality:
    def __init__(self, df_param, constraint_type, target_type, is_weekly_selected, convert_to_weekly_data):
        """initialization

        Args:
            df_param (dataframe): model param
        """
        df_param_opt = df_param.T
        df_param_opt.columns = df_param_opt.iloc[0, :]

        self.d_param = df_param_opt.iloc[1:, :].to_dict()

        self.constraint_type = constraint_type.lower()
        self.target_type = target_type.lower()
        self.is_weekly_selected = is_weekly_selected
        self.convert_to_weekly_data = convert_to_weekly_data
        if "cpm" in df_param.columns:
            self.use_impression = True
            self.metric = 'impression'
            if self.constraint_type == 'median':
                self.const_var = 'impression_median'
            else:
                self.const_var = 'impression_mean'
        else:
            self.use_impression = False
            self.metric = 'spend'
            if self.constraint_type == 'median':
                self.const_var = 'median spend'
            else:
                self.const_var = 'mean spend'        
            
        self.dimension_names = list(self.d_param.keys())
        # Precision used for optimization
        self.precision = 1e-0
        # Max iterations used for optimization
        self.max_iter = 5000000

    def s_curve_hill(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))
    
    def s_curve_hill_inv(self, Y, a, b, c):
        """This method performs the inverse of scurve function on param, target and
        Returns the outcome as investment"""
        Y = (Y-(self.precision/100)) if(Y==c) else Y
        if (Y<=0):
            return 0
        else:
            return ((Y * (b ** a))/(c - Y)) ** (1/a)
        

    def s_curve_hill_seasonality(
        self,
        X,
        a,
        b,
        c,
        wcoeff,
        mcoeff,
        weekday,
        month,
    ):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y with considering daily and monthly seasonality"""
        return (
            c * (X ** a / (X ** a + b ** a))
            + wcoeff[0] * weekday[0]
            + wcoeff[1] * weekday[1]
            + wcoeff[2] * weekday[2]
            + wcoeff[3] * weekday[3]
            + wcoeff[4] * weekday[4]
            + wcoeff[5] * weekday[5]
            + mcoeff[0] * month[0]
            + mcoeff[1] * month[1]
            + mcoeff[2] * month[2]
            + mcoeff[3] * month[3]
            + mcoeff[4] * month[4]
            + mcoeff[5] * month[5]
            + mcoeff[6] * month[6]
            + mcoeff[7] * month[7]
            + mcoeff[8] * month[8]
            + mcoeff[9] * month[9]
            + mcoeff[10] * month[10]
        )
         

    def s_curve_hill_inv_seasonality(
        self,
        Y,
        a,
        b,
        c,
        wcoeff,
        mcoeff,
        weekday,
        month,
    ):
        """This method performs the inverse of scurve function on param, target and considering daily and monthly seasonality
        Returns the outcome as investment"""
        Y_ = Y - (wcoeff[0] * weekday[0]
                + wcoeff[1] * weekday[1]
                + wcoeff[2] * weekday[2]
                + wcoeff[3] * weekday[3]
                + wcoeff[4] * weekday[4]
                + wcoeff[5] * weekday[5]
                + mcoeff[0] * month[0]
                + mcoeff[1] * month[1]
                + mcoeff[2] * month[2]
                + mcoeff[3] * month[3]
                + mcoeff[4] * month[4]
                + mcoeff[5] * month[5]
                + mcoeff[6] * month[6]
                + mcoeff[7] * month[7]
                + mcoeff[8] * month[8]
                + mcoeff[9] * month[9]
                + mcoeff[10] * month[10]
                )
        Y_ = (Y_-(self.precision/100)) if(Y_==c) else Y_
        if (Y_<=0):
            return 0
        else:
            return ((Y_ * (b ** a))/(c - Y_)) ** (1/a)

    
    def ini_start_value(self, df_grp, dimension_bound):
        """initialization of initial metric (spend or impression) to overcome the local minima for each dimension
        
        Returns:
            Array - float value:
                For impression: Minimum impression and corresponding spend for each dimension
                For spend: Minimum spend for each dimension
        """
        oldSpendVec = {}
        oldImpVec = {}
        
        for dim in self.dimension_names:
            if self.use_impression:
                df_grp_tmp_imp = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['impression'])!=0)].copy()
                start_value_imp = df_grp_tmp_imp[self.metric].min()
                start_value_spend=(start_value_imp*dimension_bound[dim][2])/1000
            
                input_start_imp=((dimension_bound[dim][0] * 1000) / dimension_bound[dim][2])
                input_start_spend=dimension_bound[dim][0]
                
                input_final_imp=((dimension_bound[dim][1] * 1000) / dimension_bound[dim][2])
                input_final_spend=dimension_bound[dim][1]
                
                if(input_start_spend>start_value_spend):
                    oldImpVec[dim]=input_start_imp
                    oldSpendVec[dim]=input_start_spend
                elif(input_final_spend<start_value_spend):
                    oldImpVec[dim]=input_start_imp
                    oldSpendVec[dim]=input_start_spend
                else:
                    oldImpVec[dim]=start_value_imp
                    oldSpendVec[dim]=start_value_spend
            else:
                df_grp_tmp_spend = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['spend'])!=0)].copy()
                start_value_spend = df_grp_tmp_spend['spend'].min()
                
                input_start_spend=dimension_bound[dim][0]
                input_final_spend=dimension_bound[dim][1]
            
                if(input_start_spend>start_value_spend):
                    oldSpendVec[dim]=input_start_spend
                elif(input_final_spend<start_value_spend):
                    oldSpendVec[dim]=input_start_spend
                else:
                    oldSpendVec[dim]=start_value_spend
                
        if self.use_impression:
            return oldSpendVec, oldImpVec
        else:
            return oldSpendVec
        

    def increment_factor(self, df_grp):
        """Increment value for each iteration
        
        Returns:
            Float value: Increment factor - always based on spend (irrespective of metric chosen)
        """
        # inc_factor =  df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby('date').agg({'spend':'sum','target':'sum'})['spend'].median()
        # increment = round(inc_budget*0.075)
        # increment = round(df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby(['dimension']).agg({'spend':'median'})['spend'].median())
        inc_factor = round(df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby(['dimension']).agg({'spend':self.constraint_type})['spend'].min())
        increment = round(inc_factor*0.50)
        return increment
    
    
    def initial_conversion(self, oldMetricVec):
        """initialization of initial conversions for each dimension for initail slected metric (spend or impression)
        
        Returns:
            Array - float value:
                Conversions
        """
        oldReturn = {}
        for dim in self.dimension_names:
            oldReturn[dim]=(self.s_curve_hill(oldMetricVec[dim],
                                          self.d_param[dim]["param a"],
                                          self.d_param[dim]["param b"],
                                          self.d_param[dim]["param c"]))
        return oldReturn
    

    def get_conversion_dimension(self, newSpendVec, dimension_bound, increment, newImpVec):
        """Function to get dimension and their conversion for increment budget - to derive dimension having maximum conversion
        
        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension
                totalReturn: Conversion for allocated budget for each dimension          
        """
        incReturns = {}
        incBudget = {}
            
        for dim in self.dimension_names:

            oldSpendTemp = newSpendVec[dim]
            if self.use_impression:
                oldImpTemp = newImpVec[dim]  
                        
            # check if spend allocated to a dimension + increment is less or equal to max constarint and get incremental converstions
            if((newSpendVec[dim] + increment)<=dimension_bound[dim][1]):
                incBudget[dim] = increment
            # check if spend allocated to a dimension + increment is greater than max constarint and get converstions for remaining budget for that dimension
            elif((newSpendVec[dim]<dimension_bound[dim][1]) & ((newSpendVec[dim] + increment)>dimension_bound[dim][1])):
                # getting remaining increment budget if post increment allocation budget exceeds max bound for a dimension
                incBudget[dim] = dimension_bound[dim][1] - newSpendVec[dim]
            # if max budget is exhausted for that dimension
            else:
                incBudget[dim]=0
                incReturns[dim]=-1
                continue
        
            # updated spend post increment budget allocation
            newSpendTemp = newSpendVec[dim] + incBudget[dim]

            # check for increment return
            if self.use_impression:
                newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                incReturns[dim]=(self.s_curve_hill(newImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                            -  self.s_curve_hill(oldImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            else:
                incReturns[dim]=(self.s_curve_hill(newSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                            -  self.s_curve_hill(oldSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))  

        return incReturns, incBudget
    
    
    def conversion_optimize_spend(self, increment, oldSpendVec, oldReturn, returnGoal, dimension_bound):
        """function for calculating target conversion when metric as spend is selected
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend and Conversion for every dimension
                Iterration Result: Number of iterations and corresponding spend and conversion to reach target conversion (result not used in the UI)
            Message 4001: Optimum conversion reached
            Message 4002: Exceeded number of iterations, solution couldn't be found
        """
        results_itr_df=pd.DataFrame(columns=['spend', 'return'])
        
        newSpendVec = oldSpendVec.copy()
        totalSpend = sum(oldSpendVec.values())
        totalReturn = oldReturn.copy()

        if self.use_impression:
            newImpVec = oldImpVec.copy()
        else:
            newImpVec = {}
        
        result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
        results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
        results_itr_df=results_itr_df.reset_index(drop=True)
        
        iteration=0
        itr_calibrate=0
        calibrate_flag=0
        msg=4001
        
        while(returnGoal>sum(totalReturn.values())):                   
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, newImpVec)    
            dim_idx=max(incReturns, key=incReturns.get)
            
            if(incReturns[dim_idx]>0):
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                totalReturn[dim_idx] = self.s_curve_hill(newSpendVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                returnGoal_err_per = (sum(totalReturn.values()) - returnGoal) * 100 / returnGoal
                # print("Error rate ",sum(totalReturn.values())," ",returnGoal," ",returnGoal_err_per)
                
                if((returnGoal_err_per>=-0.5) and (returnGoal_err_per<=1.5)):
                    msg=4001
                    iteration+=1
                    # print("inc - cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    break
                    
                elif((returnGoal_err_per>1.5) and (itr_calibrate<=500)):
                    itr_calibrate+=1
                    # print("not inc - cal - greater 1.5%"," ",dim_idx," ",increment)
                    newSpendVec[dim_idx] = newSpendVec[dim_idx] - incBudget[dim_idx]
                    totalReturn[dim_idx] = self.s_curve_hill(newSpendVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    calibrate_increment=round(increment*0.1)
                    increment=increment-calibrate_increment
                    
                elif(itr_calibrate>500):
                    msg=4003
                    # print("not inc - cal - cal not possible"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    raise Exception("Optimal solution not found")
                    break
                
                else:
                    msg=4001
                    iteration+=1
                    # print("inc - not cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                
            elif(incReturns[dim_idx]==-1):
                iteration+=1
                itr_calibrate+=1
                # print("not inc - increment calibartion"," ",increment)
                calibrate_increment=round(increment*0.1)
                increment=increment-calibrate_increment
            
            if(iteration>=self.max_iter):
                msg=4002
                raise Exception("Optimal solution not found")
                break
        
        newSpendVec, totalReturn, newImpVec = self.adjust_budget(newSpendVec, totalReturn, dimension_bound, None)
        # print(iteration," ",itr_calibrate)
        conversion_return_df = pd.DataFrame(totalReturn.items())
        budget_return_df = pd.DataFrame(newSpendVec.items())
        
        conversion_return_df.rename({0: 'dimension', 1: 'return'}, axis=1, inplace=True)
        budget_return_df.rename({0: 'dimension', 1: 'spend'}, axis=1, inplace=True)
        result_df = pd.merge(budget_return_df, conversion_return_df, on='dimension', how='outer')

        return result_df, results_itr_df, msg
    
    
    def conversion_optimize_impression(self, increment, oldSpendVec, oldImpVec, oldReturn, returnGoal, dimension_bound):
        """function for calculating target conversion when metric as impression is selected
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend, Impression and Conversion for every dimension
                Iterration Result: Number of iterations and corresponding spend, impression and conversion to reach target conversion (result not used in the UI)
            Message 4001: Optimum conversion reached
            Message 4002: Exceeded number of iterations, solution couldn't be found
        """
        results_itr_df=pd.DataFrame(columns=['spend', 'impression', 'return'])
        
        newSpendVec = oldSpendVec.copy()
        newImpVec = oldImpVec.copy()
        
        totalSpend = sum(oldSpendVec.values())
        totalImp = sum(oldImpVec.values())
        totalReturn = oldReturn.copy()

        if self.use_impression:
            newImpVec = oldImpVec.copy()
        else:
            newImpVec = {}
        
        result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
        results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
        results_itr_df=results_itr_df.reset_index(drop=True)
        
        iteration=0
        itr_calibrate=0
        msg=4001
        while(returnGoal>sum(totalReturn.values())):
            
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, newImpVec)    
            dim_idx=max(incReturns, key=incReturns.get)
            
            if(incReturns[dim_idx]>0):
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                newImpVec[dim_idx] = ((newSpendVec[dim_idx]*1000)/(dimension_bound[dim_idx][2]))
                totalReturn[dim_idx] = self.s_curve_hill(newImpVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                
                returnGoal_err_per = (sum(totalReturn.values()) - returnGoal) * 100 / returnGoal
                # print("Error rate ",sum(totalReturn.values())," ",returnGoal," ",returnGoal_err_per)
                
                if((returnGoal_err_per>=-0.5) and (returnGoal_err_per<=1.5)):
                    msg=4001
                    iteration+=1
                    # print("inc - cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    break
                    
                elif((returnGoal_err_per>1.5) and (itr_calibrate<=500)):
                    itr_calibrate+=1
                    # print("not inc - cal - greater 1.5%"," ",dim_idx," ",increment)
                    newSpendVec[dim_idx] = newSpendVec[dim_idx] - incBudget[dim_idx]
                    newImpVec[dim_idx] = ((newSpendVec[dim_idx]*1000)/(dimension_bound[dim_idx][2]))
                    totalReturn[dim_idx] = self.s_curve_hill(newImpVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    calibrate_increment=round(increment*0.1)
                    increment=increment-calibrate_increment
                    
                elif(itr_calibrate>500):
                    msg=4003
                    # print("not inc - cal - cal not possible"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    raise Exception("Optimal solution not found")
                    break
                
                else:
                    msg=4001
                    iteration+=1
                    # print("inc - not cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
            
            elif(incReturns[dim_idx]==-1):
                iteration+=1
                itr_calibrate+=1
                # print("not inc - increment calibartion"," ",increment)
                calibrate_increment=round(increment*0.1)
                increment=increment-calibrate_increment
            
            if(iteration>=self.max_iter):
                msg=4002
                raise Exception("Optimal solution not found")
                break
        
        # print(iteration," ",itr_calibrate)
        newSpendVec, totalReturn, newImpVec = self.adjust_budget(newSpendVec, totalReturn, dimension_bound, newImpVec)
        conversion_return_df = pd.DataFrame(totalReturn.items())
        budget_return_df = pd.DataFrame(newSpendVec.items())
        imp_return_df = pd.DataFrame(newImpVec.items())
        
        conversion_return_df.rename({0: 'dimension', 1: 'return'}, axis=1, inplace=True)
        budget_return_df.rename({0: 'dimension', 1: 'spend'}, axis=1, inplace=True)
        imp_return_df.rename({0: 'dimension', 1: 'impression'}, axis=1, inplace=True)
        
        result_df = pd.merge(imp_return_df, conversion_return_df, on='dimension', how='outer')
        result_df = pd.merge(result_df, budget_return_df, on='dimension', how='outer')

        return result_df, results_itr_df, msg
    

    def total_return(self, newSpendVec, totalReturn, dimension_bound, dim, newImpVec):
        """calculate total spend based on spend or impression
        
        Returns:
            Dictionay - 
                totalReturn: Conversion for allocated budget or impression for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        if self.use_impression:
            newImpVec[dim] = ((newSpendVec[dim]*1000)/(dimension_bound[dim][2]))
            totalReturn[dim] = self.s_curve_hill(newImpVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
        else:
            totalReturn[dim] = self.s_curve_hill(newSpendVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
        return totalReturn, newImpVec


    def adjust_budget(self, newSpendVec, totalReturn, dimension_bound_actual, newImpVec):
        """Budget for each dimension is checked and adjusted based on the following:
            Budget adjust due to rounding error in target
            If a particular dimension has zero target but some budget allocated by optimizer, this scenario occurs when inilization is done before optimization process

        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension after adjustment
                totalReturn: Conversion for allocated budget for each dimension after adjustment
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        """
        Note: No requirement for checking grouped constraints for this function
            Rounding error adjustment: Budget will remain same or floor level budget will be used
            Zero conversion dimension: Budget will be reduced to 0 or lower bound where no conversion is generated
        """
        budgetDecrement = 0

        # adjust budget due to rounding error
        for dim in self.d_param:
            dim_spend=newSpendVec[dim]
            dim_return = 0
            conv=totalReturn[dim]
            if (round(conv)>conv):
                dim_return=np.trunc(conv*10)/10
            elif (round(conv)<conv):
                dim_return=int(conv)
            else:
                continue
            dim_metric = self.s_curve_hill_inv(totalReturn[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
            
            if self.use_impression:
                dim_metric_spend=(newImpVec[dim] * dimension_bound_actual[dim][2])/1000
                if(dim_metric_spend>=dimension_bound_actual[dim][0]):
                    newImpVec[dim] = dim_metric
                    newSpendVec[dim] = dim_metric_spend
                    totalReturn[dim] = dim_return
                else:
                    continue   
            else:
                if(dim_metric>=dimension_bound_actual[dim][0]):
                    newSpendVec[dim] = dim_metric
                    totalReturn[dim] = dim_return
                else:
                    continue
            
            budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)

        # decrement unused budget from dimensions having almost zero conversion as part of budget allocation during initialization of initial budget value
        for dim in self.d_param:
            if ((totalReturn[dim]<1) and (newSpendVec[dim]>0)):
                budgetDecrement = budgetDecrement + (newSpendVec[dim] - dimension_bound_actual[dim][0])
                newSpendVec[dim] = dimension_bound_actual[dim][0]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)

        return newSpendVec, totalReturn, newImpVec
    

    def lift_cal(self, result_df, conv_goal, df_spend_dis, days, dimension_bound):
        """function for calculating different columns to be displayed for n days selected by the user
        
        Returns:
            Dataframe:
                Final Result Dataframe: Contains percentage and n days calculation
        """
        if self.use_impression:
            result_df.rename({'impression': 'recommended_impressions_for_n_days', 'spend': 'recommended_budget_for_n_days', 'return': 'estimated_return_for_n_days'}, axis=1, inplace=True)
            cpm_df=pd.DataFrame([(dim, dimension_bound[dim][2]) for dim in dimension_bound], columns=['dimension', 'cpm'])
            result_df=pd.merge(result_df, cpm_df, on='dimension', how='left')
        else:
            result_df.rename({self.metric: 'recommended_budget_for_n_days', 'return': 'estimated_return_for_n_days'}, axis=1, inplace=True)

        result_df['recommended_budget_for_n_days'] = result_df['recommended_budget_for_n_days'].round().astype(int)
        result_df['estimated_return_for_n_days'] = result_df['estimated_return_for_n_days'].round().astype(int)
        result_df['recommended_budget_per_day']=(result_df['recommended_budget_for_n_days']/days).round().astype(int)
        result_df['estimated_return_per_day']=(result_df['estimated_return_for_n_days']/days)
        result_df['estimated_return_per_day']=(result_df['estimated_return_for_n_days']/days).round()
        result_df['buget_allocation_new_%'] = (result_df['recommended_budget_for_n_days']/sum(result_df['recommended_budget_for_n_days'])).round(1)
        result_df['estimated_return_%'] = ((result_df['estimated_return_for_n_days']/sum(result_df['estimated_return_for_n_days']))*100).round(1)
        
        result_df=result_df[['dimension', 'recommended_budget_per_day', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days']]
        
        return result_df
    

    def optimizer_result_adjust(self, discard_json, df_res, df_spend_dis, dimension_bound, conv_goal, days, d_weekday, d_month, date_range, freq_type):
        """re-calculation of result based on discarded dimension budget

        Args:
            discard_json (json): key: discarded dimension ,value: spend
            df_res (dataframe): res dataframe from optimizer
            df_spend_dis (dataframe): spend distribution
            days (int): days

        Returns:
            dataframe: recal of optimizer result for discarded dimension
        """
        discard_json = {chnl:discard_json[chnl] for chnl in discard_json.keys() if(discard_json[chnl]!=0)}
        check_discard_json = bool(discard_json)
        d_dis = df_spend_dis.set_index('dimension').to_dict()['spend']

        for dim_ in discard_json.keys():
            l_append = []
            for col in df_res.columns:
                
                l_col_update = ['dimension','recommended_budget_per_day','recommended_budget_for_n_days']

                if(col in l_col_update):
                    if(col=='recommended_budget_per_day'):
                        l_append.append(int(round(discard_json[dim_])))
                    elif(col=='recommended_budget_for_n_days'):
                        l_append.append(int(round(discard_json[dim_]))*days)
                    else:
                        l_append.append(dim_)
                else:
                    l_append.append(None)

            df_res.loc[-1] = l_append
            df_res.index = df_res.index + 1  # shifting index
            df_res = df_res.sort_index()

        df_res['buget_allocation_new_%'] = ((df_res['recommended_budget_for_n_days']/sum(df_res['recommended_budget_for_n_days']))*100).round(2)

        df_res = df_res.merge(df_spend_dis[['dimension', 'median spend', 'mean spend', 'spend']], on='dimension', how='left')
        df_res['total_buget_allocation_old_%'] = ((df_res['spend']/df_res['spend'].sum())*100).round(1)

        if self.constraint_type == 'median':
            df_res['buget_allocation_old_%'] = ((df_res['median spend']/df_res['median spend'].sum())*100)
            df_res['median spend'] = df_res['median spend'].round().astype(int)
            df_res = df_res.rename(columns={"median spend": "original_constraint_budget_per_day"})
        else:
            df_res['buget_allocation_old_%'] = ((df_res['mean spend']/df_res['mean spend'].sum())*100)
            df_res['mean spend'] = df_res['mean spend'].round().astype(int)
            df_res = df_res.rename(columns={"mean spend": "original_constraint_budget_per_day"})

        for dim in self.d_param:
            budget_per_day = int(sum(df_res['recommended_budget_per_day']))
            spend_projections = budget_per_day*(df_res.loc[df_res['dimension']==dim, 'buget_allocation_old_%']/100)
            df_res.loc[df_res['dimension']==dim, 'spend_projection_constraint_for_n_day'] = spend_projections * days
            if self.use_impression:
                imp_projections = (spend_projections * 1000)/dimension_bound[dim][2]
                metric_projections = imp_projections
            else:
                metric_projections = spend_projections
            target_projection = 0
            for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both", freq=freq_type):
                day_month = str(day_.weekday())+"_"+str(day_.month)
                init_weekday = d_weekday[day_month]
                init_month = d_month[day_month]
                target_projection = target_projection + self.s_curve_hill_seasonality(metric_projections,
                                                    self.d_param[dim]["param a"],
                                                    self.d_param[dim]["param b"],
                                                    self.d_param[dim]["param c"],
                                                    list(self.d_param[dim].values())[3:9],
                                                    list(self.d_param[dim].values())[9:20],
                                                    init_weekday,
                                                    init_month)
            df_res.loc[df_res['dimension']==dim, 'current_projections_for_n_days'] = target_projection
        df_res['current_projections_for_n_days'] = df_res['current_projections_for_n_days'].round()
        df_res['current_projections_per_day'] = (df_res['current_projections_for_n_days']/days).round()
        df_res['current_projections_%'] = ((df_res['current_projections_for_n_days']/df_res['current_projections_for_n_days'].sum())*100).round(1)
        df_res['buget_allocation_old_%']=df_res['buget_allocation_old_%'].round(1)
        df_res["spend_projection_constraint_for_n_day"]=df_res["spend_projection_constraint_for_n_day"].round()
        df_res['current_projections_for_n_days']=df_res['current_projections_for_n_days'].round()

        summary_metrics_dic = {"Optimized Total Budget" : sum(df_res['recommended_budget_for_n_days'].replace({np.nan: 0}).astype(float).round().astype(int)),
                               "Optimized Total Target" : sum(df_res['estimated_return_for_n_days'].replace({np.nan: 0}).astype(float).round().astype(int)),
                               "Current Projection Total Budget" : sum(df_res["spend_projection_constraint_for_n_day"].replace({np.nan: 0}).astype(float).round().astype(int)),
                               "Current Projection Total Target" : sum(df_res['current_projections_for_n_days'].replace({np.nan: 0}).round().astype(int))
                               }
        
        if self.target_type == "revenue":
            summary_metrics_dic["Optimized CPA/ROI"] = round(summary_metrics_dic["Optimized Total Target"]/summary_metrics_dic["Optimized Total Budget"], 2)
            summary_metrics_dic["Current Projection CPA/ROI"] = round(summary_metrics_dic["Current Projection Total Target"]/summary_metrics_dic["Current Projection Total Budget"], 2)
            df_res["current_projections_CPA_ROI"] = ((df_res["current_projections_for_n_days"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["spend_projection_constraint_for_n_day"].replace({np.nan: 0, None: 0})!=0, "spend_projection_constraint_for_n_day"])).replace({np.inf: 0}).round(2)
            df_res["optimized_CPA_ROI"] = ((df_res["estimated_return_for_n_days"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["recommended_budget_for_n_days"].replace({np.nan: 0, None: 0})!=0, "recommended_budget_for_n_days"])).replace({np.inf: 0}).round(2)
        else:
            summary_metrics_dic["Optimized CPA/ROI"] = round(summary_metrics_dic["Optimized Total Budget"]/summary_metrics_dic["Optimized Total Target"], 2)
            summary_metrics_dic["Current Projection CPA/ROI"] = round(summary_metrics_dic["Current Projection Total Budget"]/summary_metrics_dic["Current Projection Total Target"], 2)
            df_res["current_projections_CPA_ROI"] = ((df_res["spend_projection_constraint_for_n_day"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["current_projections_for_n_days"].replace({np.nan: 0, None: 0})!=0, "current_projections_for_n_days"])).replace({np.inf: 0}).round(2)
            df_res["optimized_CPA_ROI"] = ((df_res["recommended_budget_for_n_days"].replace({np.nan: 0, None: 0})).div(df_res.loc[df_res["estimated_return_for_n_days"].replace({np.nan: 0, None: 0})!=0, "estimated_return_for_n_days"])).replace({np.inf: 0}).round(2)
  
        df_res["current_projections_CPA_ROI"] = np.where((df_res["spend_projection_constraint_for_n_day"]==0) | (df_res["current_projections_for_n_days"]==0), 0, df_res["current_projections_CPA_ROI"])
        df_res["optimized_CPA_ROI"] = np.where((df_res["recommended_budget_for_n_days"]==0) | (df_res["estimated_return_for_n_days"]==0), 0, df_res["optimized_CPA_ROI"])

        if self.constraint_type == 'median':
            df_res = df_res.rename(columns={"original_constraint_budget_per_day": "original_median_budget_per_day"})
            df_res = df_res.rename(columns={"buget_allocation_old_%": "median_buget_allocation_old_%"})
            df_res=df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'median_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%']]
        else:
            df_res = df_res.rename(columns={"original_constraint_budget_per_day": "original_mean_budget_per_day"})
            df_res = df_res.rename(columns={"buget_allocation_old_%": "mean_buget_allocation_old_%"})
            df_res=df_res[['dimension', 'original_mean_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'mean_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%']]

        df_res = df_res.replace({np.nan: None})

        for dim in discard_json:
            df_res.loc[df_res['dimension']==dim, 'current_projections_CPA_ROI'] = None
            df_res.loc[df_res['dimension']==dim, 'optimized_CPA_ROI'] = None

        int_cols = [i for i in df_res.columns if ((i != "dimension") & ('%' not in i) & ('CPA_ROI' not in i))]
        for i in int_cols:
            df_res.loc[df_res[i].values != None, i]=df_res.loc[df_res[i].values != None, i].astype(float).round().astype(int)

        return df_res, summary_metrics_dic, check_discard_json
    
        
    def dimension_bound_max_check(self, dimension_bound):
        for dim in dimension_bound:
            # dim_max_poss_conversion=np.floor(self.d_param[dim]["param c"]*0.95)
            dim_max_poss_conversion=int(self.d_param[dim]["param c"])
            
            dim_max_inp_budget=dimension_bound[dim][1]

            if self.use_impression:
                dim_max_inp_imp=(dim_max_inp_budget * 1000) / dimension_bound[dim][2]
                dim_max_inp_conversion=s_curve_hill(dim_max_inp_imp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_imp=int(s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                dim_max_poss_budget=(dim_max_poss_imp * dimension_bound[dim][2])/1000
            else:
                dim_max_inp_conversion=s_curve_hill(dim_max_inp_budget, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_budget=int(s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            
            if (dim_max_inp_conversion>dim_max_poss_conversion):
               dim_max_budget=dim_max_poss_budget
            elif (dim_max_inp_conversion==dim_max_poss_conversion):
               dim_max_budget=min(dim_max_poss_budget, dim_max_inp_budget)
            else:
                dim_max_budget=dim_max_inp_budget

            if(dim_max_budget<dimension_bound[dim][0]):
                dim_max_budget=dimension_bound[dim][0]
            
            dimension_bound[dim][1]=dim_max_budget
        return dimension_bound
    
    def get_seasonality_conversions(self, init_weekday, init_month):

        seasonality_conversion = 0
        for dim in self.d_param:
            wcoeff = list(self.d_param[dim].values())[3:9]
            mcoeff = list(self.d_param[dim].values())[9:20]
            seasonality_conversion_dim = (wcoeff[0] * init_weekday[0]
                                            + wcoeff[1] * init_weekday[1]
                                            + wcoeff[2] * init_weekday[2]
                                            + wcoeff[3] * init_weekday[3]
                                            + wcoeff[4] * init_weekday[4]
                                            + wcoeff[5] * init_weekday[5]
                                            + mcoeff[0] * init_month[0]
                                            + mcoeff[1] * init_month[1]
                                            + mcoeff[2] * init_month[2]
                                            + mcoeff[3] * init_month[3]
                                            + mcoeff[4] * init_month[4]
                                            + mcoeff[5] * init_month[5]
                                            + mcoeff[6] * init_month[6]
                                            + mcoeff[7] * init_month[7]
                                            + mcoeff[8] * init_month[8]
                                            + mcoeff[9] * init_month[9]
                                            + mcoeff[10] * init_month[10])
            seasonality_conversion = seasonality_conversion + seasonality_conversion_dim

        return seasonality_conversion
    
    
    def total_return_seasonality(self, Spend, dimension_bound, dim, init_weekday, init_month):
        """calculate total return based on spend or impression
        
        Returns:
                totalReturn: Conversion for allocated budget or impression for each dimension
        """
        if self.use_impression:
            Imp = ((Spend*1000)/(dimension_bound[dim][2]))
            totalReturn = self.s_curve_hill_seasonality(Imp,
                                                    self.d_param[dim]["param a"],
                                                    self.d_param[dim]["param b"],
                                                    self.d_param[dim]["param c"],
                                                    list(self.d_param[dim].values())[3:9],
                                                    list(self.d_param[dim].values())[9:20],
                                                    init_weekday,
                                                    init_month)
        else:
            totalReturn = self.s_curve_hill_seasonality(Spend,
                                                    self.d_param[dim]["param a"],
                                                    self.d_param[dim]["param b"],
                                                    self.d_param[dim]["param c"],
                                                    list(self.d_param[dim].values())[3:9],
                                                    list(self.d_param[dim].values())[9:20],
                                                    init_weekday,
                                                    init_month)
        return totalReturn

    
    def get_seasonality_result(self, result_df, dimension_bound, init_weekday, init_month):

        for dim in result_df['dimension'].unique():
            Spend = result_df[result_df['dimension']==dim]['spend'].values[0]
            result_df.loc[result_df['dimension']==dim, 'return'] = self.total_return_seasonality(Spend, dimension_bound, dim, init_weekday, init_month)
        
        return result_df
    

    def confidence_score(self, result_df, accuracy_df, df_grp, lst_dim, dimension_bound):
        """function for calculating optimization confidence score
        calculation is based on independent accuracy of each dimension (calculated using their SMAPE), % budget allocation over historic max and budget distribution across dimension based on optimization
        % budget allocation over historic max is utlized for penalizing dimensions whose recommended budget is beyond their historic maximum value (with weightage 20%)
        
        Returns:
            Value: Optimization confidence score
        """
        penalty_weightage = 0.20
        # Dimension level accuracy calculation using SMAPE (error in the model)
        accuracy_df['Accuracy'] = 1 - accuracy_df['SMAPE']
        # Maximum historic spend at dimension level
        if self.use_impression:
            max_dim_df = df_grp.groupby('dimension').agg(max_impression=('impression', 'max')).round().reset_index()
            cpm_df=pd.DataFrame([(dim, dimension_bound[dim][2]) for dim in dimension_bound], columns=['dimension', 'cpm'])
            max_dim_df = pd.merge(cpm_df, max_dim_df, on='dimension', how='left')
            max_dim_df['max_spend'] = (max_dim_df['max_impression']*max_dim_df['cpm'])/1000
            max_dim_df = max_dim_df[['dimension', 'max_spend']]
        else:
            max_dim_df = df_grp.groupby('dimension').agg(max_spend=('spend', 'max')).round().reset_index()
        score_df = pd.merge(accuracy_df[['dimension', 'Accuracy']], max_dim_df, on='dimension')
        score_df = pd.merge(score_df, result_df[['dimension', 'recommended_budget_per_day', 'buget_allocation_new_%']], on='dimension')
        # Filtering for participating dimensions in optimization (dimensions selected by the user in frontend)
        score_df = score_df[score_df['dimension'].isin(lst_dim)]
        score_df['buget_allocation_new_%'] = score_df['buget_allocation_new_%']/100

        # Calculating % budget allocation by optimization over maximum historic spend for penalising such dimensions
        score_df['budget_allocated_over_max_%'] = np.where(score_df['recommended_budget_per_day']>score_df['max_spend'],
                                                           ((score_df['recommended_budget_per_day']-score_df['max_spend'])/score_df['max_spend']), 0)
        
        # Adjusting accuracy based on above step
        # Formula used: ((Accuracy x 100%) - (% Budget Allocation over Max Spend x Penalization Weightage)) x % Recommended Budget Allocation
        score_df['score'] = (score_df['Accuracy'] - (score_df['budget_allocated_over_max_%']*penalty_weightage)) * score_df['buget_allocation_new_%']

        # Final optimization confidence score based on weighted average
        conf_score = round((sum(score_df['score'])/sum(score_df['buget_allocation_new_%']))*100, 1)

        return conf_score

                
    def execute(self, df_grp, conv_goal, date_range, df_spend_dis, discard_json, dimension_bound, lst_dim, df_score_final):
        """main function for calculating target conversion
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend/Impression and Conversion for every dimension
        """
        if self.convert_to_weekly_data == True:
            self.is_weekly_selected = True
        d_param_old=copy.deepcopy(self.d_param)
        df_param_temp= pd.DataFrame(self.d_param).T.reset_index(drop=False).rename({'index':'dimension'}, axis=1)
        df_param_temp=df_param_temp[df_param_temp['dimension'].isin(lst_dim)].reset_index(drop=True)
        df_param_opt = df_param_temp.T
        df_param_opt.columns = df_param_opt.iloc[0, :]
        d_param = df_param_opt.iloc[1:, :].to_dict()
        self.d_param=d_param

        dimension_bound_old=copy.deepcopy(dimension_bound)
        dim_bnd_temp= pd.DataFrame(dimension_bound).T.reset_index(drop=False).rename({'index':'dimension'}, axis=1)
        dim_bnd_temp=dim_bnd_temp[dim_bnd_temp['dimension'].isin(lst_dim)].reset_index(drop=True)
        dim_bnd_opt = dim_bnd_temp.T
        dim_bnd_opt.columns = dim_bnd_opt.iloc[0, :]
        dimension_bound = dim_bnd_opt.iloc[1:, :].to_dict()

        self.dimension_names = list(self.d_param.keys())
        
        dimension_bound = self.dimension_bound_max_check(dimension_bound)

        days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1
        if self.is_weekly_selected == True:
            days = int(days/7)
            day_name = pd.to_datetime(date_range[0]).day_name()[0:3]
            freq_type = "W-"+day_name
        else:
            freq_type = "D"

        init_weekday = [0, 0, 0, 0, 0, 0]
        init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d_weekday = {}
        d_month = {}
        count_day = 1
        sol = {}

        seasonality_combination = []
        seasonality_count = {}

        for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both", freq=freq_type):
            day_counter = str(day_.weekday())+"_"+str(day_.month)
            seasonality_combination = seasonality_combination + [day_counter]
            if day_counter in seasonality_count.keys():
                seasonality_count[day_counter] += 1
            else:
                seasonality_count[day_counter] = 1
        seasonality_combination = set(seasonality_combination)

        overall_seas_conv = 0
        for day_month in seasonality_combination:

            weekday = int(day_month.split('_')[0])
            month = int(day_month.split('_')[1])
                
            if weekday != 0:
                init_weekday[weekday - 1] = 1

            if month != 1:
                init_month[month - 2] = 1

            d_weekday[day_month] = init_weekday
            d_month[day_month] = init_month

            overall_seas_conv = (overall_seas_conv
                                + (self.get_seasonality_conversions(init_weekday, init_month)
                                    * seasonality_count[day_month]))

            init_weekday = [0, 0, 0, 0, 0, 0]
            init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count_day += 1            
        
        conv_goal_wo_seas = conv_goal - overall_seas_conv
        returnGoal = np.round((conv_goal_wo_seas/days),2)      
        increment = self.increment_factor(df_grp)
        # increment = round(inc_budget*0.075)

        if self.use_impression:
            oldSpendVec, oldImpVec = self.ini_start_value(df_grp, dimension_bound)
            oldReturn = self.initial_conversion(oldImpVec)
            result_df_, result_itr_df, msg = self.conversion_optimize_impression(increment, oldSpendVec, oldImpVec, oldReturn, returnGoal, dimension_bound)
            result_df_=result_df_[['dimension', 'spend', 'impression', 'return']]
            result_df_[['spend', 'impression', 'return']]=result_df_[['spend', 'impression', 'return']].round(2)
        else:
            oldSpendVec = self.ini_start_value(df_grp, dimension_bound)
            oldReturn = self.initial_conversion(oldSpendVec)
            result_df_, result_itr_df, msg = self.conversion_optimize_spend(increment, oldSpendVec, oldReturn, returnGoal, dimension_bound)
            result_df_=result_df_[['dimension', 'spend', 'return']]
            result_df_[['spend', 'return']]=result_df_[['spend', 'return']].round(2)

        for day_month in seasonality_combination:
            
            weekday = int(day_month.split('_')[0])
            month = int(day_month.split('_')[1])
                
            if weekday != 0:
                init_weekday[weekday - 1] = 1

            if month != 1:
                init_month[month - 2] = 1

            d_weekday[day_month] = init_weekday
            d_month[day_month] = init_month

            result_df_seasonality = self.get_seasonality_result(result_df_, dimension_bound, init_weekday, init_month)
        
            sol[day_month] = result_df_seasonality.set_index('dimension').T.to_dict('dict')

            init_weekday = [0, 0, 0, 0, 0, 0]
            init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count_day += 1

        # Aggregating results for entire date range
        if self.use_impression:
            result_df = pd.DataFrame(columns=['spend', 'impression', 'return'], index=self.dimension_names).fillna(0)
        else:
            result_df = pd.DataFrame(columns=['spend', 'return'], index=self.dimension_names).fillna(0)

        for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both", freq=freq_type):
            day_month = str(day_.weekday())+"_"+str(day_.month)
            temp_df = pd.DataFrame(sol[day_month]).T
            result_df = result_df.add(temp_df, fill_value=0)
        result_df = result_df.rename_axis('dimension').reset_index()

        result_df = self.lift_cal(result_df, conv_goal, df_spend_dis, days, dimension_bound)
        result_df, summary_metrics_dic, check_discard_json = self.optimizer_result_adjust(discard_json, result_df, df_spend_dis, dimension_bound, conv_goal, days, d_weekday, d_month, date_range, freq_type)        
        result_itr_df=result_itr_df.round(2)

        # Optimization confidence score calculation
        optimization_conf_score = self.confidence_score(result_df, df_score_final, df_grp, lst_dim, dimension_bound)

        return result_df, summary_metrics_dic, optimization_conf_score, check_discard_json