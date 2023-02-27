import pandas as pd
import numpy as np
import math
import copy
import warnings
warnings.filterwarnings('ignore')

# isolate function
def dimension_bound(df_param, dimension_data, constraint_type):

    """bounds for optimizer

    Returns:
        dictionary: key dimension value [min,max] / [min,max,cpm]
        dictionary: key group dimension value [list of sub-dimension] and [min,max]
        flag: multiple dimensions selected or not
    """

    constraint_type = constraint_type.lower()
    threshold = [0, 3]

    df_param_opt = df_param.T
    df_param_opt.columns = df_param_opt.iloc[0, :]

    d_param = df_param_opt.iloc[1:, :].to_dict()

    dim_bound = {}
    grp_dim_bound = {}

    if "cpm" in df_param.columns:

        if constraint_type == "median":
            const_var = "impression_median"
        else:
            const_var = "impression_mean"

        for dim in d_param.keys():
            dim_bound[dim] = [
                int(
                    (d_param[dim][const_var] * d_param[dim]["cpm"] / 1000)
                    * threshold[0]
                ),
                int(
                    (d_param[dim][const_var] * d_param[dim]["cpm"] / 1000)
                    * threshold[1]
                ),
                d_param[dim][const_var] * d_param[dim]["cpm"] / 1000,
                -20,
                20,
                round(d_param[dim]["cpm"], 2),
             ]
    else:
        for dim in d_param.keys():

            if constraint_type == "median":
                const_var = "median spend"
            else:
                const_var = "mean spend"

            dim_bound[dim] = [
                int(d_param[dim][const_var] * threshold[0]),
                int(d_param[dim][const_var] * threshold[1]),
                d_param[dim][const_var],
                -20,
                20
            ]

    grp_dim_flag = True if (len(dimension_data.keys())>1) else False
    
    if(grp_dim_flag == True):
        grp_dim_list = dimension_data[list(dimension_data.keys())[0]]
        for grp_dim in grp_dim_list:
            sub_dim_list = list({dim for dim, value in dim_bound.items() if dim.startswith(grp_dim)})
            if not(sub_dim_list):
                continue
            sub_dim_bound_min = sum([dim_bound[dim][0] for dim in sub_dim_list])
            sub_dim_bound_max = sum([dim_bound[dim][1] for dim in sub_dim_list])
            grp_dim_bound[grp_dim] = {'sub_dimension' : sub_dim_list,
                                      'constraints':[sub_dim_bound_min, sub_dim_bound_max]}

    return dim_bound, grp_dim_bound, grp_dim_flag


class optimizer_iterative:
    def __init__(self, df_param, constraint_type):
        """initialization

        Args:
            df_param (dataframe): model param
        """
        df_param_opt = df_param.T
        df_param_opt.columns = df_param_opt.iloc[0, :]

        self.d_param = df_param_opt.iloc[1:, :].to_dict()

        self.constraint_type = constraint_type.lower()

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
        """This method performs the scurve function on param, X and
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
    
    
    def dimension_bound_max_check(self, dimension_bound):
        """Restricting dimensions budget to max conversion budget if enetered budget is greater for any dimension
        
        Returns:
            Dictionary: dimension_bound
        """
        for dim in dimension_bound:
            
            # Max conversion possible
            conv=round(self.d_param[dim]["param c"])
            if conv>self.d_param[dim]["param c"]:
                dim_max_poss_conversion=(np.trunc(self.d_param[dim]["param c"]*10)/10)
            elif conv<=self.d_param[dim]["param c"]:
                dim_max_poss_conversion=int(self.d_param[dim]["param c"])
            
            # Max budget entered by user
            dim_max_inp_budget=dimension_bound[dim][1]
            
            # Geting budget for Max conversion possible and conversion for Max budget entered by user
            if self.use_impression:
                dim_max_inp_imp=(dim_max_inp_budget * 1000) / dimension_bound[dim][2]
                dim_max_inp_conversion=self.s_curve_hill(dim_max_inp_imp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_imp=int(self.s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                dim_max_poss_budget=(dim_max_poss_imp * dimension_bound[dim][2])/1000
            else:
                dim_max_inp_conversion=self.s_curve_hill(dim_max_inp_budget, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_budget=int(self.s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            
            # Comparing max conversion/budget possible and max budget/conversion entered by user
            if (dim_max_inp_budget>dim_max_poss_budget):
                dim_max_budget=dim_max_poss_budget
            elif (dim_max_inp_conversion==dim_max_poss_conversion):
                dim_max_budget=min(dim_max_poss_budget, dim_max_inp_budget)
            else:
                dim_max_budget=dim_max_inp_budget
                
            # Comparing max and min budget
            if(dim_max_budget<dimension_bound[dim][0]):
                dim_max_budget=dimension_bound[dim][0]
            
            dimension_bound[dim][1]=dim_max_budget
        return dimension_bound
    
    
    def ini_start_value(self, df_grp, dimension_bound, increment):
        """initialization of initial metric (spend or impression) to overcome the local minima for each dimension
        
        Returns:
            Array - float value:
                For impression: Minimum impression and corresponding spend for each dimension
                For spend: Minimum spend for each dimension
        
        To initialize start value:
        Getting min non-zero spend/impression from historic data for dimension whose inflection point is greater than increment
        Compare min spend from historic data and entered by user
        Compare min spend from historic data and max spend entered by user
        """
        oldSpendVec = {}
        oldImpVec = {}        

        for dim in self.dimension_names:
            if self.use_impression:
                if(((self.d_param[dim]['param b']*dimension_bound[dim][2])/1000)>increment):
                    df_grp_tmp_imp = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['impression'])!=0)].copy()
                    start_value_imp = df_grp_tmp_imp[self.metric].min()
                    start_value_spend=(start_value_imp*dimension_bound[dim][2])/1000
                else:
                    start_value_imp = 0
                    start_value_spend = 0
            
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
                if(self.d_param[dim]['param b']>increment):
                    df_grp_tmp_spend = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['spend'])!=0)].copy()
                    start_value_spend = df_grp_tmp_spend['spend'].min()
                else:
                    start_value_spend = 0
                
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
                newSpendTemp = newSpendVec[dim] + incBudget[dim]
                
                if self.use_impression:
                    newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                    incReturns[dim]=(self.s_curve_hill(newImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                             -  self.s_curve_hill(oldImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                else:
                    incReturns[dim]=(self.s_curve_hill(newSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                             -  self.s_curve_hill(oldSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            
            # check if spend allocated to a dimension + increment is greater than max constarint and get converstions for remaining budget for that dimension
            elif((newSpendVec[dim]<dimension_bound[dim][1]) & ((newSpendVec[dim] + increment)>dimension_bound[dim][1])):
                
                incBudget[dim] = dimension_bound[dim][1] - newSpendVec[dim]
                newSpendTemp = newSpendVec[dim] + incBudget[dim]
                
                if self.use_impression:
                    newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                    incReturns[dim]=(self.s_curve_hill(newImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                             -  self.s_curve_hill(oldImpTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                else:
                    incReturns[dim]=(self.s_curve_hill(newSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                             -  self.s_curve_hill(oldSpendTemp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            
            # if max budget is exhausted for that dimension
            else:
                newSpendTemp = newSpendVec[dim]
                incBudget[dim]=0
                incReturns[dim]=-1
        
        return incReturns, incBudget
        
        
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
    
    
    def allocate_remaining_budget(self, budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, iteration, msg, newImpVec):
        """allocate remaining budget when total maximum possible conversion have been allocated

        Returns:
        Dictionay - 
            newSpendVec: Budget allocated to each dimension
            totalReturn: Conversion for allocated budget for each dimension
            newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        Value - msg: Whether allocation was sucessfull or max iteration was reached
        """
        while(math.isclose(budgetGoal, sum(newSpendVec.values()), abs_tol=self.precision)!=True):

            budgetAllocate = budgetGoal-sum(newSpendVec.values())

            allocation_dim_list = []
            newSpendVec_filtered = {}
            allocationSpendVec = {}

            # Get list of dimensions which have not exhausted their entire budget
            for dim in self.d_param:
                if (newSpendVec[dim]<dimension_bound_actual[dim][1]):
                    allocation_dim_list = allocation_dim_list + [dim]
                    newSpendVec_filtered[dim] = newSpendVec[dim]

            # Check list if all remianing dimensions have been allocated some budget or not in optimization process
            check_noSpendDim = all(value == 0 for value in newSpendVec_filtered.values())

            # Get proportion to allocate remaining budget: 
            # budget allocated during optimization process (or median/mean spend if no budget is allocated in optimization process)
            
            if self.use_impression:
                agg_constSpend=sum((self.d_param[dim_filtered][self.const_var]*self.d_param[dim_filtered]['cpm'])/1000 for dim_filtered in allocation_dim_list)
            else:
                agg_constSpend=sum(self.d_param[dim_filtered][self.const_var] for dim_filtered in allocation_dim_list)
            
            for dim in allocation_dim_list:
                incrementProportion = 0
                budgetRemainDim = 0
                incrementCalibration = 0

                if (check_noSpendDim == True):
                    if self.use_impression:
                        allocationSpendVec[dim] = ((self.d_param[dim][self.const_var]*self.d_param[dim]['cpm'])/1000)/agg_constSpend
                    else:
                        allocationSpendVec[dim] = self.d_param[dim][self.const_var]/agg_constSpend
                else:
                    allocationSpendVec[dim] = newSpendVec_filtered[dim]/sum(newSpendVec_filtered.values())

                incrementProportion = budgetAllocate * allocationSpendVec[dim]
                budgetRemainDim = dimension_bound_actual[dim][1] - newSpendVec[dim]

                if (budgetRemainDim>=incrementProportion):
                    incrementCalibration = incrementProportion
                else:
                    incrementCalibration = budgetRemainDim

                newSpendVec[dim] = newSpendVec[dim] + incrementCalibration

                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)

            iteration+=1
            
            if (iteration>self.max_iter):
                msg = 4002
                raise Exception("Optimal solution not found")

        return newSpendVec, totalReturn, msg, newImpVec
        
        
    def projections_compare(self, newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, newImpVec):
        """Budget for each dimension is checked and adjusted based on comparing with historic budget projection,
            if the budget allocation by optimization is higher for same target

        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension after adjustment
                totalReturn: Conversion for allocated budget for each dimension after adjustment
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        budgetDecrement = 0

        # Comparing budget allocation and target with current projections and adjusting budget if required
        if self.use_impression:
            d_const_spend = {dim: ((value[self.const_var] * dimension_bound_actual[dim][2])/1000) for dim, value in self.d_param.items()}
        else:
            d_const_spend = {dim: value[self.const_var] for dim, value in self.d_param.items()}

        agg_const_spend = sum({value for dim, value in d_const_spend.items()})
        const_spend_per = {dim:value/agg_const_spend for dim, value in d_const_spend.items()}

        spend_projection = {}
        imp_projection = {}
        return_projection = {}

        for dim in self.d_param:

            dim_spend = 0
            spend_projection[dim] = budgetGoal * const_spend_per[dim]
            return_projection, imp_projection = self.total_return(spend_projection, return_projection, dimension_bound_actual, dim, imp_projection)
            
            if((round(return_projection[dim])>=self.d_param[dim]['param c']) | (round(return_projection[dim])==0)):
                continue

            dim_metric_estimate = int(self.s_curve_hill_inv(round(return_projection[dim]), self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            if self.use_impression:
                dim_imp_estimate = dim_metric_estimate
                dim_spend_estimate = (dim_imp_estimate * dimension_bound_actual[dim][2])/1000
            else:
                dim_spend_estimate = dim_metric_estimate
            dim_spend = min(spend_projection[dim], dim_spend_estimate)

            if ((dim_spend>=dimension_bound_actual[dim][0]) and (dim_spend<=dimension_bound_actual[dim][1])):
                if ((round(totalReturn[dim])==round(return_projection[dim])) and (newSpendVec[dim]>dim_spend)):
                    budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)
                    newSpendVec[dim] = dim_spend
                    totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)

        return newSpendVec, totalReturn, newImpVec


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
        budgetDecrement = 0

        # adjust budget due to rounding error
        for dim in self.d_param:
            dim_spend=newSpendVec[dim]
            conv=totalReturn[dim]
            if (round(conv)>conv):
                totalReturn[dim]=np.trunc(conv*10)/10
            elif (round(conv)<conv):
                totalReturn[dim]==int(conv)
            else:
                continue
            dim_metric = self.s_curve_hill_inv(totalReturn[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
            if self.use_impression:
                newImpVec[dim] = dim_metric
                newSpendVec[dim] = (newImpVec[dim] * dimension_bound_actual[dim][2])/1000
            else:
                newSpendVec[dim] = dim_metric
            budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)

        # decrement unused budget from dimensions having almost zero conversion as part of budget allocation during initialization of initial budget value
        for dim in self.d_param:
            if ((totalReturn[dim]<1) and (newSpendVec[dim]>0)):
                budgetDecrement = budgetDecrement + (newSpendVec[dim] - dimension_bound_actual[dim][0])
                newSpendVec[dim] = dimension_bound_actual[dim][0]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)
    
        return newSpendVec, totalReturn, newImpVec


    def budget_optimize(self, increment_factor, oldSpendVec, oldReturn, budgetGoal, dimension_bound, dimension_bound_actual, oldImpVec):
        """function for calculating budget when metric as spend is selected
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend and Conversion for every dimension
                Iterration Result: Number of iterations and corresponding spend and conversion to reach budget (df not used in the UI)
            Message:
                4001: Optimum budget reached
                4002: Exceeded number of iterations, optimal solution couldn't be found
        """
        resultIter_df=pd.DataFrame(columns=['spend', 'impression', 'return'])
        
        newSpendVec = oldSpendVec.copy()
        totalSpend = sum(oldSpendVec.values())
        totalReturn = oldReturn.copy()
              
        if self.use_impression:
            newImpVec = oldImpVec.copy()
        else:
            newImpVec = {}
        
        resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
        
        increment = increment_factor
        iteration = 0
        msg = 4001
        check_noConversion = 0

        while(budgetGoal > sum(newSpendVec.values())):
            
            # Get dim with max incremental conversion
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, newImpVec)  
            dim_idx = max(incReturns, key=incReturns.get)
            
            # If incremental conversion is present
            if(incReturns[dim_idx] > 0):
                iteration+=1
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound, dim_idx, newImpVec)
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)

             # If incremental conversion is not present
            else:
                newSpendVec, totalReturn, msg, newImpVec = self.allocate_remaining_budget(budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, iteration, msg, newImpVec)
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
                break
                
            # If budget goal is reached, check for dimension with no conversion but some budget is allocated during inital spend allocation
            if(math.isclose(sum(newSpendVec.values()), budgetGoal, abs_tol=self.precision)):
                if (check_noConversion == 0):
                    check_noConversion = 1
                    newSpendVec, totalReturn, newImpVec = self.projections_compare(newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, newImpVec)
                    newSpendVec, totalReturn, newImpVec = self.adjust_budget(newSpendVec, totalReturn, dimension_bound_actual, newImpVec)
                    increment = increment_factor
                    resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
                else:
                    break
            
            # If allocated budget exceed budget goal
            elif(sum(newSpendVec.values())>budgetGoal):
                newSpendVec[dim_idx] = newSpendVec[dim_idx] - incBudget[dim_idx]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim_idx, newImpVec)
                increment = budgetGoal - sum(newSpendVec.values())
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)

            # If max iteration is reached
            if(iteration > self.max_iter):
                msg = 4002
                raise Exception("Optimal solution not found")
                            
        conversion_return_df = pd.DataFrame(totalReturn.items())
        budget_return_df = pd.DataFrame(newSpendVec.items())
        
        conversion_return_df.rename({0: 'dimension', 1: 'return'}, axis=1, inplace=True)
        budget_return_df.rename({0: 'dimension', 1: 'spend'}, axis=1, inplace=True)
        result_df = pd.merge(budget_return_df, conversion_return_df, on='dimension', how='outer')
                   
        if self.use_impression:
            imp_return_df = pd.DataFrame(newImpVec.items())
            imp_return_df.rename({0: 'dimension', 1: 'impression'}, axis=1, inplace=True)
            result_df = pd.merge(result_df, imp_return_df, on='dimension', how='outer')
        else:
            resultIter_df=resultIter_df[['spend', 'return']]
                
        return result_df, resultIter_df, msg
    
        
    def lift_cal(self, result_df, budget_per_day, df_spend_dis, days, dimension_bound):
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
        result_df['buget_allocation_new_%'] = (result_df['recommended_budget_per_day']/sum(result_df['recommended_budget_per_day'])).round(2)
        result_df['estimated_return_%'] = ((result_df['estimated_return_per_day']/sum(result_df['estimated_return_per_day']))*100).round(2)
                
        result_df=result_df[['dimension', 'recommended_budget_per_day', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days']]
        
        return result_df
    
    
    def optimizer_result_adjust(self, discard_json, df_res, df_spend_dis, dimension_bound, budget_per_day, days):
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

        df_res['buget_allocation_new_%'] = ((df_res['recommended_budget_per_day']/sum(df_res['recommended_budget_per_day']))*100).round(2)

        df_res = df_res.merge(df_spend_dis[['dimension', 'median spend', 'mean spend', 'spend']], on='dimension', how='left')
        # df_res['buget_allocation_old_%'] = ((df_res['spend']/df_res['spend'].sum())*100).round(2)

        if self.constraint_type == 'median':
            df_res['buget_allocation_old_%'] = ((df_res['median spend']/df_res['median spend'].sum())*100)
            df_res['median spend'] = df_res['median spend'].round().astype(int)
            df_res = df_res.rename(columns={"median spend": "original_median_budget_per_day"})
        else:
            df_res['buget_allocation_old_%'] = ((df_res['mean spend']/df_res['mean spend'].sum())*100)
            df_res['mean spend'] = df_res['mean spend'].round().astype(int)
            df_res = df_res.rename(columns={"mean spend": "original_mean_budget_per_day"})

        for dim in self.d_param:
            spend_projections = budget_per_day*(df_res.loc[df_res['dimension']==dim, 'buget_allocation_old_%']/100)
            if self.use_impression:
                imp_projections = (spend_projections * 1000)/dimension_bound[dim][2]
                metric_projections = imp_projections
            else:
                metric_projections = spend_projections
            df_res.loc[df_res['dimension']==dim, 'current_projections_per_day'] = self.s_curve_hill(metric_projections, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]).round(2)
        df_res['current_projections_for_n_days'] = df_res['current_projections_per_day']*days
        df_res['current_projections_%'] = ((df_res['current_projections_per_day']/df_res['current_projections_per_day'].sum())*100)
        df_res['buget_allocation_old_%']=df_res['buget_allocation_old_%'].round(2)
        df_res = df_res.replace({np.nan: None})

        if self.constraint_type == 'median':
            df_res=df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days', 'current_projections_for_n_days']]
        else:
            df_res=df_res[['dimension', 'original_mean_budget_per_day', 'recommended_budget_per_day', 'buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days', 'current_projections_for_n_days']]
        
        int_cols = [i for i in df_res.columns if ((i != "dimension") & ('%' not in i))]
        for i in int_cols:
            df_res.loc[df_res[i].values != None, i]=df_res.loc[df_res[i].values != None, i].astype(float).round().astype(int)

        return df_res
    
    
    def check_initialization_required(self, increment, df_grp, dimension_bound, dimension_bound_actual, budget_per_day):
        """Initialization to avoid local minima problem-
                Budget: Minimum non-zero spend based on historic data is intialized
                Target, impression (if selected): Respective target, impression based on spend is initialized 
                Note: If sum initailized spend exceeds budget per day then initialization is done based on user constarints
            
        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension
                totalReturn: Conversion for allocated budget for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        
        if self.use_impression:
            oldSpendVec_initial, oldImpVec_initial = self.ini_start_value(df_grp, dimension_bound, increment)
            oldReturn_initial = self.initial_conversion(oldImpVec_initial)
            initial_investment = sum(oldReturn_initial.values())
            
            oldSpendVec_input = {dim:value[0] for dim, value in dimension_bound.items()}
            oldImpVec_input = {dim:((oldSpendVec_input[dim]*1000)/(dimension_bound[dim][2])) for dim in self.dimension_names}
            oldReturn_input = {dim:(self.s_curve_hill(oldImpVec_input[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])) for dim in self.dimension_names}
            
            inflection_spend_dim = {dim:((self.d_param[dim]['param b']*dimension_bound[dim][2])/1000) for dim in self.dimension_names}
            const_spend_dim = {dim:((self.d_param[dim][self.const_var]*dimension_bound[dim][2])/1000) for dim in self.dimension_names}
            threshold = np.mean(list({const_spend_dim[dim] for dim in self.dimension_names if inflection_spend_dim[dim] > increment }))              
            
            if((initial_investment<=budget_per_day) and (budget_per_day>threshold)):
                oldSpendVec = copy.deepcopy(oldSpendVec_initial)
                oldImpVec = copy.deepcopy(oldImpVec_initial)
                oldReturn = copy.deepcopy(oldReturn_initial)
            else:
                oldSpendVec = copy.deepcopy(oldSpendVec_input)
                oldImpVec = copy.deepcopy(oldImpVec_input)
                oldReturn = copy.deepcopy(oldReturn_input)
                
        else:
            oldSpendVec_initial = self.ini_start_value(df_grp, dimension_bound, increment)
            oldReturn_initial = self.initial_conversion(oldSpendVec_initial)
            initial_investment = sum(oldSpendVec_initial.values())
            
            oldSpendVec_input = {dim:value[0] for dim, value in dimension_bound.items()}
            oldReturn_input = {dim:(self.s_curve_hill(oldSpendVec_input[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])) for dim in self.dimension_names}
            threshold = np.mean(list({value[self.const_var] for dim, value in self.d_param.items() if value['param b'] > increment }))

            if((initial_investment<=budget_per_day) and (budget_per_day>threshold)):
                oldSpendVec = copy.deepcopy(oldSpendVec_initial)
                oldReturn = copy.deepcopy(oldReturn_initial)
            else:
                oldSpendVec = copy.deepcopy(oldSpendVec_input)
                oldReturn = copy.deepcopy(oldReturn_input)
            oldImpVec = None
            
        return oldSpendVec, oldReturn, oldImpVec
            
                
    def execute(self, df_grp, budget, days, df_spend_dis, discard_json, dimension_bound):
        """main function for calculating target conversion
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend/Impression and Conversion for every dimension
        """
        # Restricting dimensions budget to max conversion budget if enetered budget is greater for any dimension
        dimension_bound_actual = copy.deepcopy(dimension_bound)
        dimension_bound = self.dimension_bound_max_check(dimension_bound)
        
        # Considering budget per day till 2 decimal points: truncting (and not rounding-off)
        budget_per_day = budget/days
        budget_per_day = (np.trunc(budget_per_day*100)/100)
        # budget_per_day = np.round((budget/days),2)
                   
        # Calculating increment budget for optimization
        increment = self.increment_factor(df_grp)
        
        """optimization process-
            Initialization if required
            Optimzation on budget and constarints
        """
        oldSpendVec, oldReturn, oldImpVec = self.check_initialization_required(increment, df_grp, dimension_bound, dimension_bound_actual, budget_per_day)
        if self.use_impression:
            result_df, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, oldImpVec)
            result_df=result_df[['dimension', 'spend', 'impression', 'return']]
            result_df[['spend', 'impression', 'return']]=result_df[['spend', 'impression', 'return']].round(2)
        else:
            result_df, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, None)
            result_df=result_df[['dimension', 'spend', 'return']]
            result_df[['spend', 'return']]=result_df[['spend', 'return']].round(2)

        # Calculating other variables for optimization plan for front end
        result_df=self.lift_cal(result_df, budget_per_day, df_spend_dis, days, dimension_bound)
        result_df=self.optimizer_result_adjust(discard_json, result_df, df_spend_dis, dimension_bound_actual, budget_per_day, days)
        
        # Df for iterative steps, not displayed in front end
        result_itr_df=result_itr_df.round(2)

        return result_df


class optimizer_iterative_seasonality:
    def __init__(self, df_param, constraint_type):
        """initialization

        Args:
            df_param (dataframe): model param
        """
        df_param_opt = df_param.T
        df_param_opt.columns = df_param_opt.iloc[0, :]

        self.d_param = df_param_opt.iloc[1:, :].to_dict()

        self.constraint_type = constraint_type.lower()

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

    
    def s_curve_hill(
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
        

    def s_curve_hill_(self, X, a, b, c):
        """This method performs the scurve function on param, X and
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


    def s_curve_hill_inv_seas(
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
    
    
    def dimension_bound_max_check(self, dimension_bound):
        """Restricting dimensions budget to max conversion budget if enetered budget is greater for any dimension
        
        Returns:
            Dictionary: dimension_bound
        """
        for dim in dimension_bound:
            
            # Max conversion possible
            conv=round(self.d_param[dim]["param c"])
            if conv>self.d_param[dim]["param c"]:
                dim_max_poss_conversion=(np.trunc(self.d_param[dim]["param c"]*10)/10)
            elif conv<=self.d_param[dim]["param c"]:
                dim_max_poss_conversion=int(self.d_param[dim]["param c"])
            
            # Max budget entered by user
            dim_max_inp_budget=dimension_bound[dim][1]
            
            # Geting budget for Max conversion possible and conversion for Max budget entered by user
            if self.use_impression:
                dim_max_inp_imp=(dim_max_inp_budget * 1000) / dimension_bound[dim][2]
                dim_max_inp_conversion=self.s_curve_hill_(dim_max_inp_imp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_imp=int(self.s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                dim_max_poss_budget=(dim_max_poss_imp * dimension_bound[dim][2])/1000
            else:
                dim_max_inp_conversion=self.s_curve_hill_(dim_max_inp_budget, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_budget=int(self.s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            
            # Comparing max conversion/budget possible and max budget/conversion entered by user
            if (dim_max_inp_budget>dim_max_poss_budget):
                dim_max_budget=dim_max_poss_budget
            elif (dim_max_inp_conversion==dim_max_poss_conversion):
                dim_max_budget=min(dim_max_poss_budget, dim_max_inp_budget)
            else:
                dim_max_budget=dim_max_inp_budget
                
            # Comparing max and min budget
            if(dim_max_budget<dimension_bound[dim][0]):
                dim_max_budget=dimension_bound[dim][0]
            
            dimension_bound[dim][1]=dim_max_budget
        return dimension_bound
    
    
    def ini_start_value(self, df_grp, dimension_bound, increment):
        """initialization of initial metric (spend or impression) to overcome the local minima for each dimension
        
        Returns:
            Array - float value:
                For impression: Minimum impression and corresponding spend for each dimension
                For spend: Minimum spend for each dimension
        
        To initialize start value:
        Getting min non-zero spend/impression from historic data for dimension whose inflection point is greater than increment
        Compare min spend from historic data and entered by user
        Compare min spend from historic data and max spend entered by user
        """
        oldSpendVec = {}
        oldImpVec = {}        

        for dim in self.dimension_names:
            if self.use_impression:
                if(((self.d_param[dim]['param b']*dimension_bound[dim][2])/1000)>increment):
                    df_grp_tmp_imp = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['impression'])!=0)].copy()
                    start_value_imp = df_grp_tmp_imp[self.metric].min()
                    start_value_spend=(start_value_imp*dimension_bound[dim][2])/1000
                else:
                    start_value_imp = 0
                    start_value_spend = 0
            
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
                if(self.d_param[dim]['param b']>increment):
                    df_grp_tmp_spend = df_grp[(df_grp['dimension']==dim) & (np.floor(df_grp['spend'])!=0)].copy()
                    start_value_spend = df_grp_tmp_spend['spend'].min()
                else:
                    start_value_spend = 0
                
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
    
    
    def initial_conversion(self, oldMetricVec, init_weekday, init_month):
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
                                                self.d_param[dim]["param c"],
                                                list(self.d_param[dim].values())[3:9],
                                                list(self.d_param[dim].values())[9:20],
                                                init_weekday,
                                                init_month))

        return oldReturn

    
    def get_conversion_dimension(self, newSpendVec, dimension_bound, increment, init_weekday, init_month, newImpVec):
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
                newSpendTemp = newSpendVec[dim] + incBudget[dim]
                
                if self.use_impression:
                    newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                    incReturns[dim]=(self.s_curve_hill(newImpTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month)
                                -  self.s_curve_hill(oldImpTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month))
                else:
                    incReturns[dim]=(self.s_curve_hill(newSpendTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month)
                                -  self.s_curve_hill(oldSpendTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month))
            
            # check if spend allocated to a dimension + increment is greater than max constarint and get converstions for remaining budget for that dimension
            elif((newSpendVec[dim]<dimension_bound[dim][1]) & ((newSpendVec[dim] + increment)>dimension_bound[dim][1])):
                
                incBudget[dim] = dimension_bound[dim][1] - newSpendVec[dim]
                newSpendTemp = newSpendVec[dim] + incBudget[dim]
                
                if self.use_impression:
                    newImpTemp = ((newSpendTemp*1000)/(dimension_bound[dim][2]))
                    incReturns[dim]=(self.s_curve_hill(newImpTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month)
                                -  self.s_curve_hill(oldImpTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month))
                else:
                    incReturns[dim]=(self.s_curve_hill(newSpendTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month)
                                -  self.s_curve_hill(oldSpendTemp,
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month))

            # if max budget is exhausted for that dimension
            else:
                newSpendTemp = newSpendVec[dim]
                incBudget[dim]=0
                incReturns[dim]=-1
        
        return incReturns, incBudget
        
        
    def total_return(self, newSpendVec, totalReturn, dimension_bound, dim, init_weekday, init_month, newImpVec):
        """calculate total spend based on spend or impression
        
        Returns:
            Dictionay - 
                totalReturn: Conversion for allocated budget or impression for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        if self.use_impression:
            newImpVec[dim] = ((newSpendVec[dim]*1000)/(dimension_bound[dim][2]))
            totalReturn[dim] = self.s_curve_hill(newImpVec[dim],
                                                    self.d_param[dim]["param a"],
                                                    self.d_param[dim]["param b"],
                                                    self.d_param[dim]["param c"],
                                                    list(self.d_param[dim].values())[3:9],
                                                    list(self.d_param[dim].values())[9:20],
                                                    init_weekday,
                                                    init_month)
        else:
            totalReturn[dim] = self.s_curve_hill(newSpendVec[dim],
                                                    self.d_param[dim]["param a"],
                                                    self.d_param[dim]["param b"],
                                                    self.d_param[dim]["param c"],
                                                    list(self.d_param[dim].values())[3:9],
                                                    list(self.d_param[dim].values())[9:20],
                                                    init_weekday,
                                                    init_month)
        return totalReturn, newImpVec
    
    
    def allocate_remaining_budget(self, budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, iteration, msg, init_weekday, init_month, newImpVec):
        """allocate remaining budget when total maximum possible conversion have been allocated

        Returns:
        Dictionay - 
            newSpendVec: Budget allocated to each dimension
            totalReturn: Conversion for allocated budget for each dimension
            newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        Value - msg: Whether allocation was sucessfull or max iteration was reached
        """
        while(math.isclose(budgetGoal, sum(newSpendVec.values()), abs_tol=self.precision)!=True):

            budgetAllocate = budgetGoal-sum(newSpendVec.values())

            allocation_dim_list = []
            newSpendVec_filtered = {}
            allocationSpendVec = {}

            # Get list of dimensions which have not exhausted their entire budget
            for dim in self.d_param:
                if (newSpendVec[dim]<dimension_bound_actual[dim][1]):
                    allocation_dim_list = allocation_dim_list + [dim]
                    newSpendVec_filtered[dim] = newSpendVec[dim]

            # Check list if all remianing dimensions have been allocated some budget or not in optimization process
            check_noSpendDim = all(value == 0 for value in newSpendVec_filtered.values())

            # Get proportion to allocate remaining budget: 
            # budget allocated during optimization process (or median/mean spend if no budget is allocated in optimization process)
            if self.use_impression:
                agg_constSpend=sum((self.d_param[dim_filtered][self.const_var]*self.d_param[dim_filtered]['cpm'])/1000 for dim_filtered in allocation_dim_list)
            else:
                agg_constSpend=sum(self.d_param[dim_filtered][self.const_var] for dim_filtered in allocation_dim_list)
            
            for dim in allocation_dim_list:
                incrementProportion = 0
                budgetRemainDim = 0
                incrementCalibration = 0

                if (check_noSpendDim == True):
                    if self.use_impression:
                        allocationSpendVec[dim] = ((self.d_param[dim][self.const_var]*self.d_param[dim]['cpm'])/1000)/agg_constSpend
                    else:
                        allocationSpendVec[dim] = self.d_param[dim][self.const_var]/agg_constSpend
                else:
                    allocationSpendVec[dim] = newSpendVec_filtered[dim]/sum(newSpendVec_filtered.values())

                incrementProportion = budgetAllocate * allocationSpendVec[dim]
                budgetRemainDim = dimension_bound_actual[dim][1] - newSpendVec[dim]

                if (budgetRemainDim>=incrementProportion):
                    incrementCalibration = incrementProportion
                else:
                    incrementCalibration = budgetRemainDim

                newSpendVec[dim] = newSpendVec[dim] + incrementCalibration

                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, init_weekday, init_month, newImpVec)

            iteration+=1
            
            if (iteration>self.max_iter):
                msg = 4002
                raise Exception("Optimal solution not found")

        return newSpendVec, totalReturn, msg, newImpVec


    def projections_compare(self, newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, init_weekday, init_month,  newImpVec):
        """Budget for each dimension is checked and adjusted based on comparing with historic budget projection,
            if the budget allocation by optimization is higher for same target

        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension after adjustment
                totalReturn: Conversion for allocated budget for each dimension after adjustment
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        budgetDecrement = 0

        # Comparing budget allocation and target with current projections and adjusting budget if required
        if self.use_impression:
            d_const_spend = {dim: ((value[self.const_var] * dimension_bound_actual[dim][2])/1000) for dim, value in self.d_param.items()}
        else:
            d_const_spend = {dim: value[self.const_var] for dim, value in self.d_param.items()}

        agg_const_spend = sum({value for dim, value in d_const_spend.items()})
        const_spend_per = {dim:value/agg_const_spend for dim, value in d_const_spend.items()}

        spend_projection = {}
        imp_projection = {}
        return_projection = {}

        for dim in self.d_param:

            dim_spend = 0
            spend_projection[dim] = budgetGoal * const_spend_per[dim]
            return_projection, imp_projection = self.total_return(spend_projection, return_projection, dimension_bound_actual, dim, init_weekday, init_month, imp_projection)

            # calculating max conversion for dimension with considering daily and monthly seasonality
            wcoeff = list(self.d_param[dim].values())[3:9]
            mcoeff = list(self.d_param[dim].values())[9:20]
            max_conversion_dim = (self.d_param[dim]['param c']
            + wcoeff[0] * init_weekday[0]
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

            if((round(return_projection[dim])>=max_conversion_dim) | (round(return_projection[dim])==0)):
                continue

            dim_metric_estimate = int(self.s_curve_hill_inv_seas(round(return_projection[dim]),
                                                                        self.d_param[dim]["param a"],
                                                                        self.d_param[dim]["param b"],
                                                                        self.d_param[dim]["param c"],
                                                                        list(self.d_param[dim].values())[3:9],
                                                                        list(self.d_param[dim].values())[9:20],
                                                                        init_weekday,
                                                                        init_month))
            if self.use_impression:
                dim_imp_estimate = dim_metric_estimate
                dim_spend_estimate = (dim_imp_estimate * dimension_bound_actual[dim][2])/1000
            else:
                dim_spend_estimate = dim_metric_estimate
            dim_spend = min(spend_projection[dim], dim_spend_estimate)

            if ((dim_spend>=dimension_bound_actual[dim][0]) and (dim_spend<=dimension_bound_actual[dim][1])):
                if ((round(totalReturn[dim])==round(return_projection[dim])) and (newSpendVec[dim]>dim_spend)):
                    budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)
                    newSpendVec[dim] = dim_spend
                    totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, init_weekday, init_month, newImpVec)

        return newSpendVec, totalReturn, newImpVec


    def adjust_budget(self, newSpendVec, totalReturn, dimension_bound_actual, init_weekday, init_month,  newImpVec):
        """Budget for each dimension is checked and adjusted based on the following:
            Budget adjust due to rounding error in target
            If a particular dimension has zero target but some budget allocated by optimizer, this scenario occurs when inilization is done before optimization process

        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension after adjustment
                totalReturn: Conversion for allocated budget for each dimension after adjustment
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        budgetDecrement = 0

        # adjust budget due to rounding error
        for dim in self.d_param:
            dim_spend=newSpendVec[dim]
            conv=totalReturn[dim]
            if (round(conv)>conv):
                totalReturn[dim]=(np.trunc(conv*10)/10)
            elif (round(conv)<conv):
                totalReturn[dim]==int(conv)
            else:
                continue
            dim_metric = self.s_curve_hill_inv_seas(totalReturn[dim],
                                                    self.d_param[dim]["param a"],
                                                    self.d_param[dim]["param b"],
                                                    self.d_param[dim]["param c"],
                                                    list(self.d_param[dim].values())[3:9],
                                                    list(self.d_param[dim].values())[9:20],
                                                    init_weekday,
                                                    init_month)
            if self.use_impression:
                newImpVec[dim] = dim_metric
                newSpendVec[dim] = (newImpVec[dim] * dimension_bound_actual[dim][2])/1000
            else:
                newSpendVec[dim] = dim_metric
            budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)

        # decrement unused budget from dimensions having almost zero conversion as part of budget allocation during initialization of initial budget value
        for dim in self.d_param:
            if ((totalReturn[dim]<1) and (newSpendVec[dim]>0)):
                budgetDecrement = budgetDecrement + (newSpendVec[dim] - dimension_bound_actual[dim][0])
                newSpendVec[dim] = dimension_bound_actual[dim][0]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, init_weekday, init_month, newImpVec)

        # add seasonlaity related target when spend is 0
        for dim in self.d_param:
            if (newSpendVec[dim]==0):
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, init_weekday, init_month, newImpVec)

        return newSpendVec, totalReturn, newImpVec
              
    
    def budget_optimize(self, increment_factor, oldSpendVec, oldReturn, budgetGoal, dimension_bound, dimension_bound_actual, init_weekday, init_month, oldImpVec):
        """function for calculating budget when metric as spend is selected
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend and Conversion for every dimension
                Iterration Result: Number of iterations and corresponding spend and conversion to reach budget (df not used in the UI)
            Message:
                4001: Optimum budget reached
                4002: Exceeded number of iterations, optimal solution couldn't be found
        """
        resultIter_df=pd.DataFrame(columns=['spend', 'impression', 'return'])
        
        newSpendVec = oldSpendVec.copy()
        totalSpend = sum(oldSpendVec.values())
        totalReturn = oldReturn.copy()
              
        if self.use_impression:
            newImpVec = oldImpVec.copy()
        else:
            newImpVec = {}
        
        resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
        
        increment = increment_factor
        iteration = 0
        msg = 4001
        check_noConversion = 0

        while(budgetGoal > sum(newSpendVec.values())):
            
            # Get dim with max incremental conversion
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, init_weekday, init_month, newImpVec)  
            dim_idx = max(incReturns, key=incReturns.get)
            
            # If incremental conversion is present
            if(incReturns[dim_idx] > 0):
                iteration+=1
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound, dim_idx, init_weekday, init_month, newImpVec)
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)

             # If incremental conversion is not present
            else:
                newSpendVec, totalReturn, msg, newImpVec = self.allocate_remaining_budget(budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, iteration, msg, init_weekday, init_month, newImpVec)
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
                break
                
            # If budget goal is reached, check for dimension with no conversion but some budget is allocated during inital spend allocation
            if(math.isclose(sum(newSpendVec.values()), budgetGoal, abs_tol=self.precision)):
                if (check_noConversion == 0):
                    check_noConversion = 1
                    newSpendVec, totalReturn, newImpVec = self.projections_compare(newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, init_weekday, init_month,  newImpVec)
                    newSpendVec, totalReturn, newImpVec = self.adjust_budget(newSpendVec, totalReturn, dimension_bound_actual, init_weekday, init_month,  newImpVec)
                    increment = increment_factor
                    resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
                else:
                    break
            
            # If allocated budget exceed budget goal
            elif(sum(newSpendVec.values())>budgetGoal):
                newSpendVec[dim_idx] = newSpendVec[dim_idx] - incBudget[dim_idx]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim_idx, init_weekday, init_month, newImpVec)
                increment = budgetGoal - sum(newSpendVec.values())
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)

            # If max iteration is reached
            if(iteration > self.max_iter):
                msg = 4002
                raise Exception("Optimal solution not found")
                
        conversion_return_df = pd.DataFrame(totalReturn.items())
        budget_return_df = pd.DataFrame(newSpendVec.items())
        
        conversion_return_df.rename({0: 'dimension', 1: 'return'}, axis=1, inplace=True)
        budget_return_df.rename({0: 'dimension', 1: 'spend'}, axis=1, inplace=True)
        result_df = pd.merge(budget_return_df, conversion_return_df, on='dimension', how='outer')
                   
        if self.use_impression:
            imp_return_df = pd.DataFrame(newImpVec.items())
            imp_return_df.rename({0: 'dimension', 1: 'impression'}, axis=1, inplace=True)
            result_df = pd.merge(result_df, imp_return_df, on='dimension', how='outer')
        else:
            resultIter_df=resultIter_df[['spend', 'return']]
                
        return result_df, resultIter_df, msg
    
        
    def lift_cal(self, result_df, budget_per_day, df_spend_dis, days, dimension_bound):
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
        result_df['estimated_return_per_day']=(result_df['estimated_return_for_n_days']/days).round().astype(int)
        result_df['buget_allocation_new_%'] = (result_df['recommended_budget_for_n_days']/sum(result_df['recommended_budget_for_n_days'])).round(2)
        result_df['estimated_return_%'] = ((result_df['estimated_return_for_n_days']/sum(result_df['estimated_return_for_n_days']))*100).round(2)
        
        result_df=result_df[['dimension', 'recommended_budget_per_day', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days']]
        
        return result_df
    
    
    def optimizer_result_adjust(self, discard_json, df_res, df_spend_dis, dimension_bound, budget_per_day, days, d_weekday, d_month, date_range):
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
        # df_res['buget_allocation_old_%'] = ((df_res['spend']/df_res['spend'].sum())*100).round(2)

        if self.constraint_type == 'median':
            df_res['buget_allocation_old_%'] = ((df_res['median spend']/df_res['median spend'].sum())*100)
            df_res['median spend'] = df_res['median spend'].round().astype(int)
            df_res = df_res.rename(columns={"median spend": "original_median_budget_per_day"})
        else:
            df_res['buget_allocation_old_%'] = ((df_res['mean spend']/df_res['mean spend'].sum())*100)
            df_res['mean spend'] = df_res['mean spend'].round().astype(int)
            df_res = df_res.rename(columns={"mean spend": "original_mean_budget_per_day"})

        for dim in self.d_param:
            spend_projections = budget_per_day*(df_res.loc[df_res['dimension']==dim, 'buget_allocation_old_%']/100)
            if self.use_impression:
                imp_projections = (spend_projections * 1000)/dimension_bound[dim][2]
                metric_projections = imp_projections
            else:
                metric_projections = spend_projections
            target_projection = 0
            for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both"):
                day_month = str(day_.weekday())+"_"+str(day_.month)
                init_weekday = d_weekday[day_month]
                init_month = d_month[day_month]
                target_projection = target_projection + self.s_curve_hill(metric_projections,
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
        df_res['current_projections_%'] = ((df_res['current_projections_for_n_days']/df_res['current_projections_for_n_days'].sum())*100)
        df_res['buget_allocation_old_%']=df_res['buget_allocation_old_%'].round(2)
        df_res = df_res.replace({np.nan: None})

        if self.constraint_type == 'median':
            df_res=df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days', 'current_projections_for_n_days']]
        else:
            df_res=df_res[['dimension', 'original_mean_budget_per_day', 'recommended_budget_per_day', 'buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days', 'current_projections_for_n_days']]
        
        int_cols = [i for i in df_res.columns if ((i != "dimension") & ('%' not in i))]
        for i in int_cols:
            df_res.loc[df_res[i].values != None, i]=df_res.loc[df_res[i].values != None, i].astype(float).round().astype(int)

        return df_res
    
    
    def check_initialization_required(self, increment, df_grp, dimension_bound, dimension_bound_actual, budget_per_day, init_weekday, init_month):
        """Initialization to avoid local minima problem-
                Budget: Minimum non-zero spend based on historic data is intialized
                Target, impression (if selected): Respective target, impression based on spend is initialized 
                Note: If sum initailized spend exceeds budget per day then initialization is done based on user constarints
            
        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension
                totalReturn: Conversion for allocated budget for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        
        if self.use_impression:
            oldSpendVec_initial, oldImpVec_initial = self.ini_start_value(df_grp, dimension_bound, increment)
            oldReturn_initial = self.initial_conversion(oldImpVec_initial, init_weekday, init_month)
            initial_investment = sum(oldReturn_initial.values())
            
            oldSpendVec_input = {dim:value[0] for dim, value in dimension_bound.items()}
            oldImpVec_input = {dim:((oldSpendVec_input[dim]*1000)/(dimension_bound[dim][2])) for dim in self.dimension_names}
            oldReturn_input = {dim:(self.s_curve_hill(oldImpVec_input[dim],
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month)) for dim in self.dimension_names}
            
            inflection_spend_dim = {dim:((self.d_param[dim]['param b']*dimension_bound[dim][2])/1000) for dim in self.dimension_names}
            const_spend_dim = {dim:((self.d_param[dim][self.const_var]*dimension_bound[dim][2])/1000) for dim in self.dimension_names}
            threshold = np.mean(list({const_spend_dim[dim] for dim in self.dimension_names if inflection_spend_dim[dim] > increment }))              
            
            if((initial_investment<=budget_per_day) and (budget_per_day>threshold)):
                oldSpendVec = copy.deepcopy(oldSpendVec_initial)
                oldImpVec = copy.deepcopy(oldImpVec_initial)
                oldReturn = copy.deepcopy(oldReturn_initial)
            else:
                oldSpendVec = copy.deepcopy(oldSpendVec_input)
                oldImpVec = copy.deepcopy(oldImpVec_input)
                oldReturn = copy.deepcopy(oldReturn_input)
                
        else:
            oldSpendVec_initial = self.ini_start_value(df_grp, dimension_bound, increment)
            oldReturn_initial = self.initial_conversion(oldSpendVec_initial, init_weekday, init_month)
            initial_investment = sum(oldSpendVec_initial.values())
            
            oldSpendVec_input = {dim:value[0] for dim, value in dimension_bound.items()}
            
            oldReturn_input = {dim:(self.s_curve_hill(oldSpendVec_input[dim],
                                                        self.d_param[dim]["param a"],
                                                        self.d_param[dim]["param b"],
                                                        self.d_param[dim]["param c"],
                                                        list(self.d_param[dim].values())[3:9],
                                                        list(self.d_param[dim].values())[9:20],
                                                        init_weekday,
                                                        init_month)) for dim in self.dimension_names}

            threshold = np.mean(list({value[self.const_var] for dim, value in self.d_param.items() if value['param b'] > increment }))

            if((initial_investment<=budget_per_day) and (budget_per_day>threshold)):
                oldSpendVec = copy.deepcopy(oldSpendVec_initial)
                oldReturn = copy.deepcopy(oldReturn_initial)
            else:
                oldSpendVec = copy.deepcopy(oldSpendVec_input)
                oldReturn = copy.deepcopy(oldReturn_input)
            oldImpVec = None
            
        return oldSpendVec, oldReturn, oldImpVec
            
                
    def execute(self, df_grp, budget, date_range, df_spend_dis, discard_json, dimension_bound):
        """main function for calculating target conversion
        
        Returns:
            Dataframe:
                Final Result Dataframe: Optimized Spend/Impression and Conversion for every dimension
        """
        # Restricting dimensions budget to max conversion budget if enetered budget is greater for any dimension
        dimension_bound_actual = copy.deepcopy(dimension_bound)
        dimension_bound = self.dimension_bound_max_check(dimension_bound)

        days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1
        
        # Considering budget per day till 2 decimal points: truncting (and not rounding-off)
        budget_per_day = budget/days
        budget_per_day = (np.trunc(budget_per_day*100)/100)
        # budget_per_day = np.round((budget/days),2)

        d_param_ = pd.DataFrame(self.d_param)
        d_param_.loc["spend_%", :] = (
            d_param_.loc["spend_%", :] / d_param_.loc["spend_%", :].sum()
        )
        self.d_param = d_param_.to_dict()

        init_weekday = [0, 0, 0, 0, 0, 0]
        init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d_weekday = {}
        d_month = {}
        count_day = 1
        sol = {}
        sol_check = {}
                   
        # Calculating increment budget for optimization
        increment = self.increment_factor(df_grp)

        # Checking distinct combination of daily and monthly for seasonality
        seasonality_combination = []
        for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both"):
            seasonality_combination = seasonality_combination + [str(day_.weekday())+"_"+str(day_.month)]
        seasonality_combination = set(seasonality_combination)

        # Optimization for each combination of seasonality
        for day_month in seasonality_combination:
            
            weekday = int(day_month.split('_')[0])
            month = int(day_month.split('_')[1])
                
            if weekday != 0:
                init_weekday[weekday - 1] = 1

            if month != 1:
                init_month[month - 2] = 1

            d_weekday[day_month] = init_weekday
            d_month[day_month] = init_month

            """optimization process-
            Initialization if required
            Optimzation on budget and constarints
            """
            oldSpendVec, oldReturn, oldImpVec = self.check_initialization_required(increment, df_grp, dimension_bound, dimension_bound_actual, budget_per_day, init_weekday, init_month)
            
            if self.use_impression:
                result_df_, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, init_weekday, init_month, oldImpVec)
                result_df_=result_df_[['dimension', 'spend', 'impression', 'return']]
                result_df_[['spend', 'impression', 'return']]=result_df_[['spend', 'impression', 'return']].round()
            else:
                result_df_, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, init_weekday, init_month, None)
                result_df_=result_df_[['dimension', 'spend', 'return']]
                result_df_[['spend', 'return']]=result_df_[['spend', 'return']].round()

            sol[day_month] = result_df_.set_index('dimension').T.to_dict('dict')
            sol_check[day_month] = msg

            init_weekday = [0, 0, 0, 0, 0, 0]
            init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count_day += 1
        
        for day_month in sol_check.keys():

            if sol_check[day_month] != 4001:
                raise Exception("Optimal solution not found")

        # Aggregating results for entire date range
        if self.use_impression:
            result_df = pd.DataFrame(columns=['spend', 'impression', 'return'], index=self.dimension_names).fillna(0)
        else:
            result_df = pd.DataFrame(columns=['spend', 'return'], index=self.dimension_names).fillna(0)

        for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both"):
            day_month = str(day_.weekday())+"_"+str(day_.month)
            temp_df = pd.DataFrame(sol[day_month]).T
            result_df = result_df.add(temp_df, fill_value=0)
        result_df = result_df.rename_axis('dimension').reset_index()
        
        # Calculating other variables for optimization plan for front end
        result_df=self.lift_cal(result_df, budget_per_day, df_spend_dis, days, dimension_bound)
        result_df=self.optimizer_result_adjust(discard_json, result_df, df_spend_dis, dimension_bound_actual, budget_per_day, days, d_weekday, d_month, date_range)
        
        # Df for iterative steps, not displayed in front end
        result_itr_df=result_itr_df.round(2)

        return result_df