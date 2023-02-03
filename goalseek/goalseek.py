import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#isolate function
def s_curve_hill_inv(Y, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return ((Y * (b ** a))/(c - Y)) ** (1/a)
    
def dimension_bound(df_param, df_grp):

    """bounds for optimizer

    Returns:
        dictionary: key dimension value [min,max] / [min,max,cpm]
    """

    threshold = [0, 3]

    df_param_opt = df_param.T
    df_param_opt.columns = df_param_opt.iloc[0, :]

    d_param = df_param_opt.iloc[1:, :].to_dict()

    dim_bound = {}

    if "cpm" in df_param.columns:
        for dim in d_param.keys():
            
            dim_min_imp=int(round(df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['impression'])!=0)]['impression'].min()))
            dim_min_imp_per=-int(round((1-(dim_min_imp/d_param[dim]["impression_median"]))*100))
            dim_bound[dim] = [
                int(dim_min_imp * d_param[dim]["cpm"] / 1000),
                int(
                    (d_param[dim]["impression_median"] * d_param[dim]["cpm"] / 1000)
                    * threshold[1]
                ),
                int(round(d_param[dim]["impression_median"] * d_param[dim]["cpm"] / 1000)),
                dim_min_imp_per,
                200,
                int(d_param[dim]["cpm"])
            ]
    else:
        for dim in d_param.keys():
            
            dim_min_budget=int(round(df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['spend'])!=0)]['spend'].min()))
            dim_min_budget_per=-int(round((1-(dim_min_budget/d_param[dim]["median spend"]))*100))
            
            dim_bound[dim] = [
                dim_min_budget,
                int(d_param[dim]["median spend"] * threshold[1]),
                int(round(d_param[dim]["median spend"])),
                dim_min_budget_per,
                200
            ]

    return dim_bound

def s_curve_hill(X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))

def conversion_bound(df_param, df_grp, df_bounds, lst_dim):

    """bounds for conversion optimizer

    Returns:
        array: max and min number conversions - max is only considered to be 95% of max conversions
    """
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
            dim_ini_imp=df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['impression'])!=0)]['impression'].min()
            dim_ini_budget=(dim_ini_imp * df_bounds[dim][2]) / 1000
            
            dim_min_inp_budget=int(df_bounds[dim][0])
            dim_max_inp_budget=int(df_bounds[dim][1])
            
            # Getting minimum conversions
            if (dim_min_inp_budget>dim_ini_budget):
                dim_min_budget=dim_min_inp_budget
            elif (dim_max_inp_budget<dim_ini_budget):
                dim_min_budget=dim_min_inp_budget
            else:
                dim_min_budget=dim_ini_budget
            dim_min_imp=(dim_min_budget * 1000) / df_bounds[dim][2]
            dim_min_conversion=s_curve_hill(dim_min_imp, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            
            # Getting maximum conversions
            # dim_max_poss_conversion=np.floor(d_param[dim]["param c"]*0.95)
            dim_max_poss_conversion=int(d_param[dim]["param c"])
            dim_max_inp_imp = (dim_max_inp_budget * 1000) / df_bounds[dim][2]
            dim_max_inp_conversion=s_curve_hill(dim_max_inp_imp, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            if (dim_max_inp_conversion>=dim_max_poss_conversion):
                dim_max_conversion=dim_max_poss_conversion
            else:
                dim_max_conversion=dim_max_inp_conversion
            
        else:
            dim_ini_budget=df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['spend'])!=0)]['spend'].min()
        
            dim_min_inp_budget=int(df_bounds[dim][0])
            dim_max_inp_budget=int(df_bounds[dim][1])
            
            # Getting minimum conversions
            if (dim_min_inp_budget>dim_ini_budget):
                dim_min_budget=dim_min_inp_budget
            elif (dim_max_inp_budget<dim_ini_budget):
                dim_min_budget=dim_min_inp_budget
            else:
                dim_min_budget=dim_ini_budget
            dim_min_conversion=s_curve_hill(dim_min_budget, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            
            # Getting maximum conversions
            # dim_max_poss_conversion=np.floor(d_param[dim]["param c"]*0.95)
            dim_max_poss_conversion=int(d_param[dim]["param c"])
            dim_max_inp_conversion=s_curve_hill(dim_max_inp_budget, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            if (dim_max_inp_conversion>=dim_max_poss_conversion):
                dim_max_conversion=dim_max_poss_conversion
            else:
                dim_max_conversion=dim_max_inp_conversion
                
        dim_conversion[dim]=[dim_min_conversion, dim_max_conversion]
        min_conversion=min_conversion+dim_conversion[dim][0]
        max_conversion=max_conversion+dim_conversion[dim][1]
            
    return [int(np.ceil(min_conversion)), int(np.floor(max_conversion))]

class optimizer_conversion:
    def __init__(self, df_param):
        """initialization

        Args:
            df_param (dataframe): model param
        """
        df_param_opt = df_param.T
        df_param_opt.columns = df_param_opt.iloc[0, :]

        self.d_param = df_param_opt.iloc[1:, :].to_dict()

        if "cpm" in df_param.columns:
            self.use_impression = True
            self.metric = 'impression'
        else:
            self.use_impression = False
            self.metric = 'spend'
            
        self.dimension_names = list(self.d_param.keys())

    def s_curve_hill(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))
    
    def s_curve_hill_inv(self, Y, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
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
#         inc_factor =  df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby('date').agg({self.metric:'sum','target':'sum'})[self.metric].median()
        inc_factor =  df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby('date').agg({'spend':'sum','target':'sum'})['spend'].median()
        # print(inc_factor*0.075)
        return inc_factor
    
    
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
#         print(oldReturn, oldMetricVec)       
        return oldReturn
    
    
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
        
        result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
        results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
        results_itr_df=results_itr_df.reset_index(drop=True)
        
        iteration=0
        itr_calibrate=0
        calibrate_flag=0
        msg=4001
        while(returnGoal>sum(totalReturn.values())):
            
            incReturns = {}
            
            for count,dim in enumerate(self.dimension_names):
                
                oldSpendTemp = newSpendVec[dim]

                if((newSpendVec[dim] + increment)<=dimension_bound[dim][1]):
                    newSpendTemp = newSpendVec[dim] + increment
                    incReturns[dim]=(self.s_curve_hill(newSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                                 -  self.s_curve_hill(oldSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                else:
                    newSpendTemp = newSpendVec[dim]
                    incReturns[dim]=-1
                
            dim_idx=max(incReturns, key=incReturns.get)
            
            if(incReturns[dim_idx]>0):
                
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + increment
                totalReturn[dim_idx] = self.s_curve_hill(newSpendVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                
                returnGoal_err_per = (sum(totalReturn.values()) - returnGoal) * 100 / returnGoal
                # print("Error rate ",sum(totalReturn.values())," ",returnGoal," ",returnGoal_err_per)
                
                if((returnGoal_err_per>=-1.5) and (returnGoal_err_per<=1.5)):
                    msg=4001
                    iteration+=1
                    # print("inc - cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    break
                    
                elif((returnGoal_err_per>1.5) and (itr_calibrate<=50)):
                    itr_calibrate+=1
                    # print("not inc - cal - greater 1.5%"," ",dim_idx," ",increment)
                    newSpendVec[dim_idx] = newSpendVec[dim_idx] - increment
                    totalReturn[dim_idx] = self.s_curve_hill(newSpendVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    calibrate_increment=round(increment*0.1)
                    increment=increment-calibrate_increment
                    
                elif(itr_calibrate>50):
                    msg=4003
                    # print("not inc - cal - cal not possible"," ",dim_idx," ",increment)
                    result_itr_dict={'spend': sum(newSpendVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
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
            
            if(iteration>=100):
                msg=4002
                break
        
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
        
        result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
        results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
        results_itr_df=results_itr_df.reset_index(drop=True)
        
        iteration=0
        itr_calibrate=0
        msg=4001
        while(returnGoal>sum(totalReturn.values())):
            
            incReturns = {}
            
            for count,dim in enumerate(self.dimension_names):
                
                oldSpendTemp = newSpendVec[dim]
                oldImpTemp = newImpVec[dim]
                
                if((newSpendVec[dim] + increment)<=dimension_bound[dim][1]):
                    newSpendTemp = newSpendVec[dim] + increment
                    newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                    incReturns[dim]=(self.s_curve_hill(newImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                                 -  self.s_curve_hill(oldImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))    
            
                else:
                    newSpendTemp = newSpendVec[dim]
                    newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                    incReturns[dim]=-1
                                  
            dim_idx=max(incReturns, key=incReturns.get)
            
            if(incReturns[dim_idx]>0):
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + increment
                newImpVec[dim_idx] = ((newSpendVec[dim_idx]*1000)/(dimension_bound[dim_idx][2]))
                totalReturn[dim_idx] = self.s_curve_hill(newImpVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                
                returnGoal_err_per = (sum(totalReturn.values()) - returnGoal) * 100 / returnGoal
                # print("Error rate ",sum(totalReturn.values())," ",returnGoal," ",returnGoal_err_per)
                
                if((returnGoal_err_per>=-1.5) and (returnGoal_err_per<=1.5)):
                    msg=4001
                    iteration+=1
                    # print("inc - cal"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
                    break
                    
                elif((returnGoal_err_per>1.5) and (itr_calibrate<=50)):
                    itr_calibrate+=1
                    # print("not inc - cal - greater 1.5%"," ",dim_idx," ",increment)
                    newSpendVec[dim_idx] = newSpendVec[dim_idx] - increment
                    newImpVec[dim_idx] = ((newSpendVec[dim_idx]*1000)/(dimension_bound[dim_idx][2]))
                    totalReturn[dim_idx] = self.s_curve_hill(newImpVec[dim_idx], self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    calibrate_increment=round(increment*0.1)
                    increment=increment-calibrate_increment
                    
                elif(itr_calibrate>50):
                    msg=4003
                    # print("not inc - cal - cal not possible"," ",dim_idx," ",increment)
                    result_itr_dict={'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}
                    results_itr_df=results_itr_df.append(result_itr_dict, ignore_index=True)
                    results_itr_df=results_itr_df.reset_index(drop=True)
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
            
            if(iteration>=100):
                msg=4002
                break
        
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
    
    def lift_cal(self, result_df, conv_goal, df_spend_dis, days, dimension_bound):
        """function for calculating different columns to be displayed for n days selected by the user
        
        Returns:
            Dataframe:
                Final Result Dataframe: Contains percentage and n days calculation
        """
        if self.use_impression:
            result_df.rename({self.metric: 'estimated_impressions_per_day', 'return': 'recommended_return_per_day'}, axis=1, inplace=True)
            cpm_df=pd.DataFrame([(dim, dimension_bound[dim][2]) for dim in dimension_bound], columns=['dimension', 'cpm'])
            result_df=pd.merge(result_df, cpm_df, on='dimension', how='left')
            result_df['estimated_budget_per_day']=((result_df['cpm']*result_df['estimated_impressions_per_day'])/1000).round().astype(int)
        else:
            result_df.rename({self.metric: 'estimated_budget_per_day', 'return': 'recommended_return_per_day'}, axis=1, inplace=True)
        
        result_df['estimated_budget_per_day']=result_df['estimated_budget_per_day'].round().astype(int)
        result_df['recommended_return_per_day']=result_df['recommended_return_per_day'].round().astype(int)
        result_df['estimated_budget_per_day']=np.where(result_df['recommended_return_per_day']<1, 0, result_df['estimated_budget_per_day'])
        result_df['recommended_return_per_day']=np.where(result_df['recommended_return_per_day']<1, 0, result_df['recommended_return_per_day'])
        result_df['estimated_budget_for_n_days'] = (result_df['estimated_budget_per_day']*days).round().astype(int)
        result_df['recommended_return_for_n_days'] = (result_df['recommended_return_per_day']*days).round().astype(int)
        result_df['buget_allocation_new_%'] = (result_df['estimated_budget_per_day']/sum(result_df['estimated_budget_per_day'])).round(2)
        result_df['recommended_return_%'] = ((result_df['recommended_return_per_day']/sum(result_df['recommended_return_per_day']))*100).round(2)
        
        result_df=pd.merge(result_df,df_spend_dis[['dimension', 'median spend', 'return_conv']],how='left',on='dimension')
        result_df['median spend']=result_df['median spend'].round().astype(int)
        result_df.rename({'median spend': 'original_median_budget_per_day', 'return_conv': 'original_agg_return_per_day'}, axis=1, inplace=True)
        result_df['original_return_%']=((result_df['original_agg_return_per_day']/sum(result_df['original_agg_return_per_day']))*100).round(2)
        
        result_df=result_df[['dimension', 'original_median_budget_per_day', 'estimated_budget_per_day', 'buget_allocation_new_%', 'estimated_budget_for_n_days', 'recommended_return_per_day', 'recommended_return_%', 'recommended_return_for_n_days', 'original_return_%']]
        
        #check for impressions logic - budget will be in form of cpm or impressions
        #check for optimizer logic as well - df_spend_dis only has median impressions do we need to add cpm as well

        return result_df
    
    
    def optimizer_result_adjust(self, discard_json, df_res, df_spend_dis, days):
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

        d_dis = df_spend_dis.set_index('dimension').to_dict()['spend']

        for dim_ in discard_json.keys():
            l_append = []
            for col in df_res.columns:
                
                l_col_update = ['dimension','estimated_budget_per_day','estimated_budget_for_n_days']

                if(col in l_col_update):
                    if(col=='estimated_budget_per_day'):
                        l_append.append(int(round(discard_json[dim_])))
                    elif(col=='estimated_budget_for_n_days'):
                        l_append.append(int(round(discard_json[dim_]))*days)
                    else:
                        l_append.append(dim_)
                else:
                    l_append.append(None)

            df_res.loc[-1] = l_append
            df_res.index = df_res.index + 1  # shifting index
            df_res = df_res.sort_index()

        df_res['buget_allocation_new_%'] = ((df_res['estimated_budget_per_day']/sum(df_res['estimated_budget_per_day']))*100).round(2)

        df_res = df_res.merge(df_spend_dis[['dimension', 'median spend', 'spend']], on='dimension', how='left')
        # df_res['buget_allocation_old_%'] = ((df_res['spend']/df_res['spend'].sum())*100).round(2)
        df_res['buget_allocation_old_%'] = ((df_res['median spend']/df_res['median spend'].sum())*100).round(2)
        
        df_res = df_res.drop('original_median_budget_per_day', axis=1)
        df_res['median spend'] = df_res['median spend'].round().astype(int)
        df_res = df_res.rename(columns={"median spend": "original_median_budget_per_day"})
        
        df_res=df_res[['dimension', 'original_median_budget_per_day', 'estimated_budget_per_day', 'buget_allocation_old_%', 'buget_allocation_new_%', 'estimated_budget_for_n_days', 'recommended_return_per_day', 'recommended_return_%', 'recommended_return_for_n_days']]
        
        int_cols = [i for i in df_res.columns if ((i != "dimension") & ('%' not in i))]
        for i in int_cols:
            df_res.loc[df_res[i].values != None, i]=df_res.loc[df_res[i].values != None, i].astype(float).round().astype(int)

        return df_res
    
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
                
    def execute(self, df_grp, conv_goal, days, df_spend_dis, discard_json, dimension_bound, lst_dim):
        """main function for calculating target conversion
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend/Impression and Conversion for every dimension
        """
        d_param_old=self.d_param
        
        df_param_temp= pd.DataFrame(self.d_param).T.reset_index(drop=False).rename({'index':'dimension'}, axis=1)
        df_param_temp=df_param_temp[df_param_temp['dimension'].isin(lst_dim)].reset_index(drop=True)
        df_param_opt = df_param_temp.T
        df_param_opt.columns = df_param_opt.iloc[0, :]
        d_param = df_param_opt.iloc[1:, :].to_dict()
        
        self.d_param=d_param
        self.dimension_names = list(self.d_param.keys())
        
        dimension_bound_old=dimension_bound
        dimension_bound={dim: dimension_bound[dim] for dim in lst_dim}
        
        dimension_bound = self.dimension_bound_max_check(dimension_bound)
        
        returnGoal = np.round((conv_goal/days),2)
                     
        inc_budget = self.increment_factor(df_grp)
        # print(inc_budget)
        increment = round(inc_budget*0.075)

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
        
        result_df=self.lift_cal(result_df, conv_goal, df_spend_dis, days, dimension_bound)
        result_df=self.optimizer_result_adjust(discard_json, result_df, df_spend_dis, days)
        
        result_itr_df=result_itr_df.round(2)
        
#         final_result = [sum(result_df['spend']), sum(result_df['return'])]
#         final_result = [sum(result_df['estimated_budget_per_day']), sum(result_df['recommended_return_per_day'])]

        return result_df, msg