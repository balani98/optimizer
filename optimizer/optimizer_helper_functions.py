import pandas as pd
import numpy as np
import math
import copy
from kneebow.rotor import Rotor
import warnings
warnings.filterwarnings('ignore')

# isolate function
def dimension_bound(df_param, dimension_data, constraint_type, is_group_dimension_selected):

    """dimension level and group level bounds for optimizer

    Returns:
        dictionary: key dimension value [min,max] for non-cpm case / [min,max,cpm] for cpm case
        dictionary: key group dimension value [list of sub-dimension] and [min,max]
    """

    # median/mean selection by the user on predict tab for bounds
    constraint_type = constraint_type.lower()

    # bound thresholds [min, max]
    threshold = [0, 3]

    df_param_opt = df_param.T
    df_param_opt.columns = df_param_opt.iloc[0, :]
    d_param = df_param_opt.iloc[1:, :].to_dict()

    dim_bound = {}
    grp_dim_bound = {}

    # calculation of dimension level bounds if impressions is selected
    # bounds calculation is based on median/mean selection, impression value and threshold
    # note: default value for percentage is assigned as [-100, 200] based on threshold
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
                -100,
                200,
                round(d_param[dim]["cpm"], 2),
             ]
            
    # calculation of dimension level bounds if impressions is not selected
    # bounds calculation is based on median/mean selection, spend value and threshold
    # note: default value for percentage is assigned as [-100, 200] based on threshold
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
                -100,
                200
            ]

    # calculation of group dimension level bounds based on dimension level bounds
    if is_group_dimension_selected == True:
        grp_dim_list = dimension_data[list(dimension_data.keys())[0]]
        for grp_dim in grp_dim_list:
            sub_dim_list = list({dim for dim, value in dim_bound.items() if dim.startswith(grp_dim)})
            if not(sub_dim_list):
                continue
            sub_dim_bound_min = sum([dim_bound[dim][0] for dim in sub_dim_list])
            sub_dim_bound_max = sum([dim_bound[dim][1] for dim in sub_dim_list])
            grp_dim_bound[grp_dim] = {'sub_dimension' : sub_dim_list,
                                    'constraints':[sub_dim_bound_min, sub_dim_bound_max],
                                    'fixed_ranges' : [sub_dim_bound_min, sub_dim_bound_max]}
        
    return dim_bound, grp_dim_bound


def investment_range(dim_bound, group_constraint, isolate_dim_list, selected_lst_dim):
    """investment range for optimizer, takes output of dimension_bound function as input
        this range is displayed in the frontend for optimization
        note: number of days/date range and discard dimension computation is handled in frontend itself

    Returns:
        list: [lower bound for investment range, upper bound for investment range]
    """
    dim_lower_bnds = 0
    dim_upper_bnds= 0

    # if no dimension is selected, returns [0, 0]
    if not selected_lst_dim:
        return [dim_lower_bnds, dim_upper_bnds]
    else:
        # if group level dimension is not selected
        if group_constraint==None:
            for dim in dim_bound:
                    dim_lower_bnds = dim_lower_bnds + dim_bound[dim][0]
                    dim_upper_bnds = dim_upper_bnds + dim_bound[dim][1]
        # if group level dimension is selected
        else:
            for dim in group_constraint:
                dim_lower_bnds = dim_lower_bnds + group_constraint[dim]['constraints'][0]
                dim_upper_bnds = dim_upper_bnds + group_constraint[dim]['constraints'][1]

            # if some dimensions are not part of any group level dimensions, they are considered seprately
            if isolate_dim_list!=None:       
                for dim in isolate_dim_list:
                    dim_lower_bnds = dim_lower_bnds + dim_bound[dim][0]
                    dim_upper_bnds = dim_upper_bnds + dim_bound[dim][1]
            
        return [dim_lower_bnds, dim_upper_bnds]


class optimizer_iterative:
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

        # Setting constraint variable based on median/mean and spend/impression selection
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
        # Setting if group dimension constraint selected or not as False, updating the flag in execute function if group dimension constraint selected is selected
        self.is_group_dimension_selected = False
        # Precision used for optimization
        self.precision = 1e-0
        # Max iterations used for optimization
        self.max_iter = 500000
        

    def s_curve_hill(self, X, a, b, c):
        """This method performs the scurve function on param, X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))
    
    
    def s_curve_hill_inv(self, Y, a, b, c):
        """This method performs the inverse of scurve function on param, target and
        Returns the outcome as investment"""
        
        # Check for Y is tending to max/saturation value of a dimension and adjusting it to avoid inf error
        Y = (Y-(self.precision/100)) if(Y==c) else Y
        if (Y<=0):
            return 0
        else:
            return ((Y * (b ** a))/(c - Y)) ** (1/a)
    
    
    def dimension_bound_max_check(self, dimension_bound):
        """Restricting dimensions budget to max conversion budget if enetered budget is greater for any dimension
        This process is performed to avoid allocating any extra amount to any dimension in the optimization process
        
        Returns:
            Dictionary: dimension_bound (with updated max bound value)
        """
        for dim in dimension_bound:
            
            # Max conversion possible
            conv=round(self.d_param[dim]["param c"])

            # Check if conv value is greater than max possible value and then adjusting it avoid inf error
            if conv>self.d_param[dim]["param c"]:
                dim_max_poss_conversion=(np.trunc(self.d_param[dim]["param c"]*10)/10)
            elif conv<=self.d_param[dim]["param c"]:
                dim_max_poss_conversion=int(self.d_param[dim]["param c"])
            
            # Max budget entered by user
            dim_max_inp_budget=dimension_bound[dim][1]
            
            # Geting budget for Max conversion possible and conversion for Max bound budget entered by user
            if self.use_impression:
                dim_max_inp_imp=(dim_max_inp_budget * 1000) / dimension_bound[dim][2]
                dim_max_inp_conversion=self.s_curve_hill(dim_max_inp_imp, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_imp=int(self.s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
                dim_max_poss_budget=(dim_max_poss_imp * dimension_bound[dim][2])/1000
            else:
                dim_max_inp_conversion=self.s_curve_hill(dim_max_inp_budget, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_max_poss_budget=int(self.s_curve_hill_inv(dim_max_poss_conversion, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]))
            
            # Comparing max possible conversion/budget and max input conversion/budget entered by user
            if (dim_max_inp_budget>dim_max_poss_budget):
                dim_max_budget=dim_max_poss_budget
            elif (dim_max_inp_conversion==dim_max_poss_conversion):
                dim_max_budget=min(dim_max_poss_budget, dim_max_inp_budget)
            else:
                dim_max_budget=dim_max_inp_budget
                
            # Comparing max and min budget in case max budget  is less than min input lower bound
            if(dim_max_budget<dimension_bound[dim][0]):
                dim_max_budget=dimension_bound[dim][0]
            
            dimension_bound[dim][1]=dim_max_budget
        return dimension_bound
    

    def transform_grouped_dimension_bound(self, dimension_bound, group_constraint, isolate_dim_list):
        """Transforming grouped dimension dictionary to use for constarints for optimization
        Allocating each sub-dimension's group-level constraint and peer sub-dimensions for code optimization (reduce one level of dictionary)

        Returns:
            Dictionary: grp_dim_bound
        """

        grp_dim_bound = {}

        # dimensions which are part of any group level dimension
        for dim in group_constraint:
            sub_dim_list = group_constraint[dim]['sub_dimension']
            sub_dim_constraint = group_constraint[dim]['constraints'][1]
            for dim_ in sub_dim_list:
                grp_dim_bound[dim_] = {'dimension' : sub_dim_list,
                                    'constraints':sub_dim_constraint}

        # dimensions which are not part of any group level dimension (unselected by the user in frontend)       
        for dim_ in isolate_dim_list:
            grp_dim_bound[dim_] = {'dimension' : [dim_],
                                    'constraints':dimension_bound[dim_][1]}
            
        return grp_dim_bound
    
    
    def ini_start_value(self, df_grp, dimension_bound, increment, grouped_dimension_bound):
        """ This function is not used anymore due to logic update
        initialization of initial metric (spend or impression) to overcome the local minima for each dimension
        
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

        if self.is_group_dimension_selected == True:
            checked_dim_list = []
            for dim in self.dimension_names:
                if dim in checked_dim_list:
                    continue
                sub_dim_list = grouped_dimension_bound[dim]['dimension']
                groupSpendConstraint = grouped_dimension_bound[dim]['constraints'] 
                agg_iniSpend=sum(oldSpendVec[dim_ini] for dim_ini in sub_dim_list)
                if(agg_iniSpend>=groupSpendConstraint):
                    for sub_dim in sub_dim_list:
                        oldSpendVec[sub_dim] = dimension_bound[sub_dim][0]
                checked_dim_list = checked_dim_list + sub_dim_list
                
        if self.use_impression:
            return oldSpendVec, oldImpVec
        else:
            return oldSpendVec
        
    
    def increment_factor(self, df_grp):
        """Increment value for each iteration
            50% of the median or mean (based on user selection) historic budget of the dimension having the minimum value is considered

        Returns:
            Float value: Increment factor - always based on spend (irrespective of metric chosen)
        """
        inc_factor = round(df_grp[df_grp['dimension'].isin(self.dimension_names)].groupby(['dimension']).agg({'spend':self.constraint_type})['spend'].min())
        increment = round(inc_factor*0.50)
        return increment
    
    
    def initial_conversion(self, oldMetricVec):
        """ This function is not used anymore due to logic update
        initialization of initial conversions for each dimension for initail slected metric (spend or impression)
        
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
    

    def get_grouped_dimension_constraint(self, grouped_dimension_bound, dim, newSpendVec):
        """Function to get aggregated allocated spend to sub-dimensions under a group (based on optimization) and the group dimension's spend constarint (based on user input)
        Returns:
            Spend variable - 
                subDimSpend: Aggregated allocated spend to each dimension
                grp_dim_const: Group dimension spend constraint         
        """
        subDimSpend = 0
        # list of sub-dimensions under a group
        sub_dim_list = grouped_dimension_bound[dim]['dimension']
        # group-level constraint (based on user input)
        grp_dim_const = grouped_dimension_bound[dim]['constraints']
        # aggregated spend allocation based on sub-dimensions under a group
        for sub_dim in sub_dim_list:
            subDimSpend = subDimSpend + newSpendVec[sub_dim]
        return subDimSpend, grp_dim_const
    

    def get_conversion_dimension(self, newSpendVec, dimension_bound, increment, grouped_dimension_bound, newImpVec):
        """Function to get dimension and their conversion for increment budget - to derive dimension having maximum conversion
        Dimension level: i) Check conversions at dimension level bounds, ii) adjust incrmental budget for dimension whose budget is about to over (if it's less than incremental budget)
        Group level (if selected by user): i) Check conversions for sub-dimensions at group level constraints, ii) adjust incrmental budget for group constraints whose budget is about to over (if it's less than incremental budget)
        
        Returns:
            Dictionay - 
                incReturns: Incremental Conversion for for each dimension   
                incBudget: Incremental Budget allocated to each dimension
        """
        incReturns = {}
        incBudget = {}
            
        for dim in self.dimension_names:

            oldSpendTemp = newSpendVec[dim]
            if self.use_impression:
                oldImpTemp = newImpVec[dim]  

            # getting sum of allocated budget to group of dimensions and grouped budget constraint
            if self.is_group_dimension_selected == True:
                subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
            
            # check if spend allocated to a dimension + increment is less or equal to max constarint and get incremental converstions
            if((newSpendVec[dim] + increment)<=dimension_bound[dim][1]):
                # checks if grouped constraints is selected
                if self.is_group_dimension_selected == True:
                    # check if post allocation of increment budget, grouped constraint is satisfied
                    if ((subDimSpend + increment)<=groupSpendConstraint):
                        incBudget[dim] = increment
                    # check if grouped constraint is lies between before and post allocation of increment budget
                    elif((subDimSpend<groupSpendConstraint) & ((subDimSpend + increment)>groupSpendConstraint)):
                        incBudget[dim] = groupSpendConstraint-subDimSpend
                    # if max budget for grouped constraint is reached
                    else:
                        incBudget[dim]=0
                        incReturns[dim]=-1
                        continue
                # if grouped constraints is not selected
                else:
                    incBudget[dim] = increment
       
            # check if spend allocated to a dimension + increment is greater than max constarint and get converstions for remaining budget for that dimension
            elif((newSpendVec[dim]<dimension_bound[dim][1]) & ((newSpendVec[dim] + increment)>dimension_bound[dim][1])):
                # getting remaining increment budget if post increment allocation budget exceeds max bound for a dimension
                temp_incBudget = dimension_bound[dim][1] - newSpendVec[dim]
                # checks if grouped constraints is selected
                if self.is_group_dimension_selected == True:
                    # check if post allocation of increment budget, grouped constraint is satisfied
                    if ((subDimSpend + temp_incBudget)<=groupSpendConstraint):
                        incBudget[dim] = temp_incBudget
                    # check if grouped constraint is lies between before and post allocation of increment budget
                    elif((subDimSpend<groupSpendConstraint) & ((subDimSpend + temp_incBudget)>groupSpendConstraint)):
                        incBudget[dim] = groupSpendConstraint-subDimSpend
                    # if max budget for grouped constraint is reached
                    else:
                        incBudget[dim]=0
                        incReturns[dim]=-1
                        continue
                # if grouped constraints is not selected
                else:
                    incBudget[dim] = temp_incBudget
            
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
    

    def get_s_curves(self, dimension_bound, df_grp):
        """get list of dimensions having shaped curves
        
        Returns:
            Dictionay - 
                totalReturn: Conversion for allocated budget or impression for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        dimList = list({dim for dim, value in self.d_param.items() if (value['param a']>1.2)})
        
        dimListFiltered_v1 = []
        dimScurveWeights = {}

        for dim in dimList:
            dim_metric = 0
            if self.use_impression:
                dim_metric = (dimension_bound[dim][1]*1000)/dimension_bound[dim][2]
                dimConv = self.s_curve_hill(dim_metric, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_metric = dim_metric/1000
            else:
                dim_metric = dimension_bound[dim][1]
                dimConv = self.s_curve_hill(dim_metric, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
            
            if (dimConv>=1):
                dimListFiltered_v1 = dimListFiltered_v1 + [dim]
                dimScurveWeights[dim] = [dim_metric, dimConv, (dimConv/dim_metric)]
                
        if self.use_impression:
            dimListFiltered_v2 = list({dim for dim in dimListFiltered_v1 if (self.d_param[dim]['param b']>(dimension_bound[dim][0]*1000)/dimension_bound[dim][2])})
        else:
            dimListFiltered_v2 = list({dim for dim in dimListFiltered_v1 if (self.d_param[dim]['param b']>dimension_bound[dim][0])})
        
        dimListFiltered = sorted(dimListFiltered_v2, key=lambda dim: dimScurveWeights[dim][2], reverse=True)

        ScurveElbowDim = {}
        ScurveElbowDim_temp = {}
        for dim in dimListFiltered:
            if self.use_impression:
                df_temp = df_grp[df_grp['dimension']==dim][['impression', 'predictions']].sort_values(by='impression').reset_index(drop=True)
                data = df_temp[df_temp['impression']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            else:
                df_temp = df_grp[df_grp['dimension']==dim][['spend', 'predictions']].sort_values(by='spend').reset_index(drop=True)
                data = df_temp[df_temp['spend']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            rotor = Rotor()
            rotor.fit_rotate(data)
            elbow_index = rotor.get_elbow_index()
            ScurveElbowDim[dim] = data[elbow_index]
            min_bnd = int(ScurveElbowDim[dim][0]*0.75)
            max_bnd = int(ScurveElbowDim[dim][0]*1.25)
            ScurveElbowDim_temp[dim] = data[elbow_index]
            counter = int((max_bnd-min_bnd)/25)
            if counter < 1:
                counter = 1
            for i in range(min_bnd, max_bnd+1, counter):
                target = self.s_curve_hill(i, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                if self.use_impression:
                    df_temp = df_temp.append({'impression':i , 'predictions' : target}, ignore_index=True).reset_index(drop=True)
                else:
                    df_temp = df_temp.append({'spend':i , 'predictions' : target}, ignore_index=True).reset_index(drop=True)
            if self.use_impression:
                df_temp = df_temp.sort_values(by='impression').reset_index(drop=True)
                data = df_temp[df_temp['impression']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            else:
                df_temp = df_temp.sort_values(by='spend').reset_index(drop=True)
                data = df_temp[df_temp['spend']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            rotor = Rotor()
            rotor.fit_rotate(data)
            elbow_index = rotor.get_elbow_index()
            ScurveElbowDim[dim] = data[elbow_index]
        # print('Scurve Elbow before: ',ScurveElbowDim_temp)
        # print('Scurve Elbow after: ',ScurveElbowDim)
        # print('Weights: ',dimScurveWeights)
        
        return dimListFiltered, dimScurveWeights, ScurveElbowDim
     
    
    def adjust_conversions(self, newSpendVec, totalReturn, dimension_bound, budgetGoal, dimScurveList, ScurveElbowDim, dimAdjustConversionPrevious, grouped_dimension_bound, newImpVec):
    
        dimCounter = dimScurveList

        dimScurveAllocationList = []
        dimAdjustConversion = []
        dimNormalSwapList = []
        dimScurveSwapList =[]
        i = 0
        # print("#####Before reinitialization: ",newSpendVec,"#####",totalReturn)
        while (i < len(dimCounter)):
            dimCheck = dimCounter[i]
            i = i + 1
            # print("Mudit", dimCheck)

            dimSpend = newSpendVec[dimCheck]
            dimConversion = totalReturn[dimCheck]
            
            dimCheckSpend = {}
            dimCheckConversion = {}

            if self.is_group_dimension_selected == True:
                subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dimCheck, newSpendVec)              

            if self.use_impression:

                dimImpression = newImpVec[dimCheck]
                dimCheckImpression = {}

                # dimCheckList = list({dim for dim, value in self.d_param.items() if ((newImpVec[dim]>dimImpression) and (newImpVec[dim]>((dimension_bound[dim][0]*1000)/dimension_bound[dimCheck][2])))})
                # dimCheckList = sorted(dimCheckList, key=lambda dim: newImpVec[dim], reverse=True)
                dimCheckList = list({dim for dim, value in self.d_param.items() if ((newSpendVec[dim]>dimSpend) and (newSpendVec[dim]>dimension_bound[dim][0]))})
                dimCheckList = sorted(dimCheckList, key=lambda dim: newSpendVec[dim], reverse=True)

                for dim_idx in dimCheckList:
                    dimSpendItr = newSpendVec[dim_idx]
                    dimImpressionItr = ((dimSpendItr*1000)/(dimension_bound[dimCheck][2]))
                    if ((dimImpressionItr>=((dimension_bound[dimCheck][0]*1000)/dimension_bound[dimCheck][2])) and (dimImpressionItr<=((dimension_bound[dimCheck][1]*1000)/dimension_bound[dimCheck][2]))):
                        tempImp = dimImpressionItr
                    elif (dimImpressionItr>=((dimension_bound[dimCheck][1]*1000)/dimension_bound[dimCheck][2])):
                        tempImp = (dimension_bound[dimCheck][1]*1000)/dimension_bound[dimCheck][2]
                    else:
                        continue

                    tempSpend = (tempImp*dimension_bound[dimCheck][2])/1000

                    if self.is_group_dimension_selected == True:
                        groupSpendAdjust = subDimSpend - dimSpend + tempSpend
                        if (groupSpendAdjust<=groupSpendConstraint):
                            tempSpend = tempSpend
                            tempImp = tempImp
                        elif((subDimSpend<groupSpendConstraint) & (groupSpendAdjust>groupSpendConstraint)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                tempSpend = tempSpend
                                tempImp = tempImp
                            elif((groupSpendConstraint-subDimSpend)>dimSpend):
                                tempSpend = groupSpendConstraint-subDimSpend
                                tempImp = ((tempSpend*1000)/(dimension_bound[dimCheck][2]))
                            else:
                                continue
                        else:
                            continue

                    tempConv = self.s_curve_hill(tempImp, self.d_param[dimCheck]["param a"], self.d_param[dimCheck]["param b"], self.d_param[dimCheck]["param c"])

                    if (tempConv > dimConversion) and (round(tempConv) >= 1):
                        if ((tempConv-dimConversion)/tempConv)>0.01:
                            dimCheckSpend[dim_idx] = tempSpend
                            dimCheckConversion[dim_idx] = tempConv
                            dimCheckImpression[dim_idx] = tempImp
                        else:
                            continue
                    else:
                        continue
                                            
                    adjustSpendSwap = dimSpend + (newSpendVec[dim_idx] - dimCheckSpend[dim_idx])
                    # print(dimSpend," ",newSpendVec[dim_idx]," ",dimCheckSpend[dim_idx])
                    adjustImpressionSwap = (adjustSpendSwap*1000)/dimension_bound[dim_idx][2]
                    adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    
                    if (adjustSpendSwap<dimension_bound[dim_idx][0]):
                        if (budgetGoal>=(sum(newSpendVec.values())+dimension_bound[dim_idx][0]-adjustSpendSwap)):
                            adjustSpendSwap = dimension_bound[dim_idx][0]
                            adjustImpressionSwap = (adjustSpendSwap*1000)/dimension_bound[dim_idx][2]
                            adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue
                    elif (adjustSpendSwap>dimension_bound[dim_idx][1]):
                        adjustSpendSwap = dimension_bound[dim_idx][1]
                        adjustImpressionSwap = (adjustSpendSwap*1000)/dimension_bound[dim_idx][2]
                        adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])

                    if self.is_group_dimension_selected == True:
                        subDimSpendSwap, groupSpendConstraintSwap = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim_idx, newSpendVec)
                        groupSpendAdjustSwap = subDimSpendSwap - newSpendVec[dim_idx] + adjustSpendSwap
                        if (groupSpendAdjustSwap<=groupSpendConstraintSwap):
                            adjustSpendSwap = adjustSpendSwap
                            adjustImpressionSwap = adjustImpressionSwap
                            adjustReturnSwap = adjustReturnSwap
                        elif((subDimSpendSwap<groupSpendConstraintSwap) & (groupSpendAdjustSwap>groupSpendConstraintSwap)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                adjustSpendSwap = adjustSpendSwap
                                adjustImpressionSwap = adjustImpressionSwap
                                adjustReturnSwap = adjustReturnSwap
                            else:
                                adjustSpendSwap = groupSpendConstraintSwap-subDimSpendSwap
                                adjustImpressionSwap = ((adjustSpendSwap*1000)/(dimension_bound[dim_idx][2]))
                                adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue

                    if (dimConversion+totalReturn[dim_idx])<(adjustReturnSwap+dimCheckConversion[dim_idx]):
                        newSpendVec[dim_idx] = adjustSpendSwap
                        newImpVec[dim_idx] = adjustImpressionSwap
                        totalReturn[dim_idx] = adjustReturnSwap
                        newSpendVec[dimCheck] = dimCheckSpend[dim_idx]
                        newImpVec[dimCheck] = dimCheckImpression[dim_idx]
                        totalReturn[dimCheck] = dimCheckConversion[dim_idx]
                        # print("Swapped ",dim_idx," ",dimCheck)
                        dimNormalSwapList = dimNormalSwapList + [dim_idx]
                        dimScurveSwapList = dimScurveSwapList + [dimCheck]
                        if (dimCheck in dimScurveList) and (newImpVec[dimCheck]>=ScurveElbowDim[dimCheck][0]):
                            dimScurveAllocationList = dimScurveAllocationList + [dimCheck]
                        break

            else:
                dimCheckList = list({dim for dim, value in self.d_param.items() if ((newSpendVec[dim]>dimSpend) and (newSpendVec[dim]>dimension_bound[dim][0]))})
                dimCheckList = sorted(dimCheckList, key=lambda dim: newSpendVec[dim], reverse=True)

                for dim_idx in dimCheckList:
                    dimSpendItr = newSpendVec[dim_idx]
                    if ((dimSpendItr>=dimension_bound[dimCheck][0]) and (dimSpendItr<=dimension_bound[dimCheck][1])):
                        tempSpend = dimSpendItr
                    elif (dimSpendItr>=dimension_bound[dimCheck][1]):
                        tempSpend = dimension_bound[dimCheck][1]
                    else:
                        continue

                    if self.is_group_dimension_selected == True:
                        groupSpendAdjust = subDimSpend - dimSpend + tempSpend
                        if (groupSpendAdjust<=groupSpendConstraint):
                            tempSpend = tempSpend
                        elif((subDimSpend<groupSpendConstraint) & (groupSpendAdjust>groupSpendConstraint)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                tempSpend = tempSpend
                            elif((groupSpendConstraint-subDimSpend)>dimSpend):
                                tempSpend = groupSpendConstraint-subDimSpend
                            else:
                                continue
                        else:
                            continue

                    tempConv = self.s_curve_hill(tempSpend, self.d_param[dimCheck]["param a"], self.d_param[dimCheck]["param b"], self.d_param[dimCheck]["param c"])
                    if (tempConv > dimConversion) and (round(tempConv) >= 1):
                        if ((tempConv-dimConversion)/tempConv)>0.01:
                            dimCheckSpend[dim_idx] = tempSpend
                            dimCheckConversion[dim_idx] = tempConv
                        else:
                            continue
                    else:
                        continue
                    
                    adjustSpendSwap = dimSpend + (newSpendVec[dim_idx] - dimCheckSpend[dim_idx])
                    adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])                        
                    
                    if (adjustSpendSwap<dimension_bound[dim_idx][0]):
                        if (budgetGoal>=(sum(newSpendVec.values())+dimension_bound[dim_idx][0]-dimSpend)):
                            adjustSpendSwap = dimension_bound[dim_idx][0]
                            adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue
                    elif (adjustSpendSwap>dimension_bound[dim_idx][1]):
                            adjustSpendSwap = dimension_bound[dim_idx][1]
                            adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])

                    if self.is_group_dimension_selected == True:
                        subDimSpendSwap, groupSpendConstraintSwap = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim_idx, newSpendVec)
                        groupSpendAdjustSwap = subDimSpendSwap - newSpendVec[dim_idx] + adjustSpendSwap
                        if (groupSpendAdjustSwap<=groupSpendConstraintSwap):
                            adjustSpendSwap = adjustSpendSwap
                            adjustReturnSwap = adjustReturnSwap
                        elif((subDimSpendSwap<groupSpendConstraintSwap) & (groupSpendAdjustSwap>groupSpendConstraintSwap)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                adjustSpendSwap = adjustSpendSwap
                                adjustReturnSwap = adjustReturnSwap
                            else:
                                adjustSpendSwap = groupSpendConstraintSwap-subDimSpendSwap
                                adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue

                    if (dimConversion+totalReturn[dim_idx])<(adjustReturnSwap+dimCheckConversion[dim_idx]):
                        newSpendVec[dim_idx] = adjustSpendSwap
                        totalReturn[dim_idx] = adjustReturnSwap
                        newSpendVec[dimCheck] = dimCheckSpend[dim_idx]
                        totalReturn[dimCheck] = dimCheckConversion[dim_idx]
                        # print("Swapped ",dim_idx," ",dimCheck)
                        dimNormalSwapList = dimNormalSwapList + [dim_idx]
                        dimScurveSwapList = dimScurveSwapList + [dimCheck]
                        if (dimCheck in dimScurveList) and (newSpendVec[dimCheck]>=ScurveElbowDim[dimCheck][0]):
                            dimScurveAllocationList = dimScurveAllocationList + [dimCheck]
                        break
        
        if dimScurveSwapList:
            dimAdjustConversion = dimScurveSwapList + dimNormalSwapList
            dimMaxAdjust = max(list({value for dim, value in newSpendVec.items() if (dim in dimScurveSwapList)}))
            # print("Value: ",dimMaxAdjust)
            
            # print("Before reinitialization: ",newSpendVec)
            for dim in newSpendVec:
                if dim in dimAdjustConversion :
                    continue
                elif dim in dimAdjustConversionPrevious:
                    continue
                elif newSpendVec[dim]>dimMaxAdjust:
                    continue
                newSpendVec[dim] = dimension_bound[dim][0]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound, dim, newImpVec)
            # print("After reinitialization: ",newSpendVec,"#####",totalReturn)
                
        return newSpendVec, totalReturn, newImpVec, set(dimScurveAllocationList)
    
    
    def allocate_remaining_budget(self, budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, grouped_dimension_bound, iteration, msg, newImpVec):
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

                if self.is_group_dimension_selected == True:
                    subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
                    if (newSpendVec[dim]<dimension_bound_actual[dim][1]) and (subDimSpend<groupSpendConstraint):
                        allocation_dim_list = allocation_dim_list + [dim]
                        newSpendVec_filtered[dim] = newSpendVec[dim]
                else:
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
        
            incrementCalibration = {}
            
            for dim in allocation_dim_list:
                incrementProportion = 0
                budgetRemainDim = 0

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
                    incrementCalibration[dim] = incrementProportion
                else:
                    incrementCalibration[dim] = budgetRemainDim

            if self.is_group_dimension_selected == True:
                for dim in allocation_dim_list:
                    sub_dim_list = grouped_dimension_bound[dim]['dimension']
                    grp_dim_inc_list = list(np.intersect1d(sub_dim_list, list(incrementCalibration.keys())))
                    agg_incSpend=sum(incrementCalibration[dim_inc] for dim_inc in grp_dim_inc_list)
                    subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
                    for grp_dim in grp_dim_inc_list:
                        if(subDimSpend+agg_incSpend>groupSpendConstraint):
                            budgetRemainDimGroup = subDimSpend+agg_incSpend-groupSpendConstraint
                            incrementCalibrationUpdate = incrementCalibration[grp_dim] - ((incrementCalibration[grp_dim]/agg_incSpend)*budgetRemainDimGroup)
                        else:
                            incrementCalibrationUpdate = incrementCalibration[grp_dim]
                        newSpendVec[grp_dim] = newSpendVec[grp_dim] + incrementCalibrationUpdate
                        totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, grp_dim, newImpVec)
                        if(grp_dim!=dim):
                            allocation_dim_list.remove(grp_dim)                
            else:
                for dim in allocation_dim_list:
                    newSpendVec[dim] = newSpendVec[dim] + incrementCalibration[dim]
                    totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)

            iteration+=1
            
            if (iteration>self.max_iter):
                msg = 4002
                # print("#####Iteration: ",iteration)
                raise Exception("Optimal solution not found")

        return newSpendVec, totalReturn, msg, newImpVec
        
        
    def projections_compare(self, newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, grouped_dimension_bound, newImpVec):
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

            if self.is_group_dimension_selected == True:
                subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
                subDimSpend_update=subDimSpend-newSpendVec[dim]+dim_spend
                if ((dim_spend>=dimension_bound_actual[dim][0]) and (dim_spend<=dimension_bound_actual[dim][1]) and (subDimSpend_update<=groupSpendConstraint)):
                    if ((round(totalReturn[dim])==round(return_projection[dim])) and (newSpendVec[dim]>dim_spend)):
                        budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)
                        newSpendVec[dim] = dim_spend
                        totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)
            else:
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


    def budget_optimize(self, increment_factor, oldSpendVec, oldReturn, budgetGoal, dimension_bound, dimension_bound_actual, grouped_dimension_bound, dimScurveList, dimScurveWeights, ScurveElbowDim, oldImpVec):
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
        check_budget_per = 0.05
        dimScurveCheck = dimScurveList
        dimAdjustConversionPrevious = []
        # print(dimScurveCheck)

        while(budgetGoal > sum(newSpendVec.values())):
            # print(sum(newSpendVec.values()))
            # Get dim with max incremental conversion
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, grouped_dimension_bound, newImpVec)  
            dim_idx = max(incReturns, key=incReturns.get)
            # print(iteration," ",dim_idx," ",incReturns[dim_idx]," ",incBudget[dim_idx])
            
            # If incremental conversion is present
            if(incReturns[dim_idx] > 0):
                iteration+=1
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound, dim_idx, newImpVec)
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)

             # If incremental conversion is not present
            else:
                budgetUntilized_temp = sum(newSpendVec.values())
                if dimScurveCheck:
                    newSpendVec, totalReturn, newImpVec, dimScurveReturn = self.adjust_conversions(newSpendVec, totalReturn, dimension_bound, budgetGoal, dimScurveCheck, ScurveElbowDim, dimAdjustConversionPrevious, grouped_dimension_bound, newImpVec)
                    dimScurveCheck = list(set(dimScurveCheck).difference(dimScurveReturn))
                    dimScurveCheck = sorted(dimScurveCheck, key=lambda dim: dimScurveWeights[dim][2], reverse=True)
                    dimAdjustConversionPrevious = list(set(dimAdjustConversionPrevious + list(dimScurveReturn)))
                    # print(dimScurveReturn)
                    # print("remaining ",dimScurveCheck)
                    # print()
                if((math.isclose(sum(newSpendVec.values()), budgetUntilized_temp, abs_tol=self.precision)) | (sum(newSpendVec.values()) > budgetUntilized_temp)):
                    newSpendVec, totalReturn, msg, newImpVec = self.allocate_remaining_budget(budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, grouped_dimension_bound, iteration, msg, newImpVec)
                    resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
                    break
                
            # If budget goal is reached, check for dimension with no conversion but some budget is allocated during inital spend allocation
            if(math.isclose(sum(newSpendVec.values()), budgetGoal, abs_tol=self.precision)):
                if (check_noConversion == 0):
                    check_noConversion = 1
                    newSpendVec, totalReturn, newImpVec = self.projections_compare(newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, grouped_dimension_bound, newImpVec)
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

            if(sum(newSpendVec.values())/budgetGoal>=check_budget_per) and (dimScurveCheck):
                # print(sum(newSpendVec.values())/budgetGoal," ",check_budget_per)
                newSpendVec, totalReturn, newImpVec, dimScurveReturn = self.adjust_conversions(newSpendVec, totalReturn, dimension_bound, budgetGoal, dimScurveCheck, ScurveElbowDim, dimAdjustConversionPrevious, grouped_dimension_bound, newImpVec)
                dimScurveCheck = list(set(dimScurveCheck).difference(dimScurveReturn))
                dimScurveCheck = sorted(dimScurveCheck, key=lambda dim: dimScurveWeights[dim][2], reverse=True)
                dimAdjustConversionPrevious = list(set(dimAdjustConversionPrevious + list(dimScurveReturn)))
                # print("#####Iteration: ",iteration)
                # print(dimScurveReturn)
                # print("remaining ",dimScurveCheck)
                # print()
                budget_per = (sum(newSpendVec.values())/budgetGoal)*100
                check_budget_per = (math.ceil(budget_per/5)*5)/100
                if check_budget_per>=0.95:
                    check_budget_per=0.95
            
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

        # print("#####Iteration: ",iteration)
                
        return result_df, resultIter_df, msg
    

    def initial_allocation(self, dimension_bound):
        """Initialization for minimum value entered by the user in the frontend in the dimension level bounds
                Budget: Minimum value entered by the user
                Target, impression (if selected): Respective target, impression based on budget is calculated 
            
        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension
                totalReturn: Conversion for allocated budget for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        if self.use_impression:
            oldSpendVec = {dim:value[0] for dim, value in dimension_bound.items()}
            oldImpVec = {dim:((oldSpendVec[dim]*1000)/(dimension_bound[dim][2])) for dim in self.dimension_names}
            oldReturn = {dim:(self.s_curve_hill(oldImpVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])) for dim in self.dimension_names}     
        else:
            oldSpendVec = {dim:value[0] for dim, value in dimension_bound.items()}
            oldReturn = {dim:(self.s_curve_hill(oldSpendVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])) for dim in self.dimension_names}
            oldImpVec = None
            
        return oldSpendVec, oldReturn, oldImpVec
    
        
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
        result_df['buget_allocation_new_%'] = (result_df['recommended_budget_per_day']/sum(result_df['recommended_budget_per_day'])).round(1)
        result_df['estimated_return_%'] = ((result_df['estimated_return_per_day']/sum(result_df['estimated_return_per_day']))*100).round(1)
                
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
            df_res=df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'median_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%', 'optimized_CPA_ROI', 'current_projections_CPA_ROI']]
        else:
            df_res = df_res.rename(columns={"original_constraint_budget_per_day": "original_mean_budget_per_day"})
            df_res = df_res.rename(columns={"buget_allocation_old_%": "mean_buget_allocation_old_%"})
            df_res=df_res[['dimension', 'original_mean_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'mean_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%', 'optimized_CPA_ROI', 'current_projections_CPA_ROI']]
     
        df_res = df_res.replace({np.nan: None})

        for dim in discard_json:
            df_res.loc[df_res['dimension']==dim, 'current_projections_CPA_ROI'] = None
            df_res.loc[df_res['dimension']==dim, 'optimized_CPA_ROI'] = None

        int_cols = [i for i in df_res.columns if ((i != "dimension") & ('%' not in i) & ('CPA_ROI' not in i))]
        for i in int_cols:
            df_res.loc[df_res[i].values != None, i]=df_res.loc[df_res[i].values != None, i].astype(float).round().astype(int)

        return df_res, summary_metrics_dic, check_discard_json
    

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
    
                
    def execute(self, df_grp, budget, days, df_spend_dis, discard_json, dimension_bound, group_constraint, isolate_dim_list, lst_dim, df_score_final):
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

        # Restricting dimensions budget to max conversion budget if enetered budget is greater for any dimension
        dimension_bound_actual = copy.deepcopy(dimension_bound)
        dimension_bound = self.dimension_bound_max_check(dimension_bound)
        
        # Considering budget per day till 2 decimal points: truncting (and not rounding-off)
        budget_per_day = budget/days
        budget_per_day = (np.trunc(budget_per_day*100)/100)
        # budget_per_day = np.round((budget/days),2)

        # Update if group dimension constraint selected
        if group_constraint!=None:
            self.is_group_dimension_selected = True

        # Transform grouped dimension dict for adding group level constarints/bounds
        if self.is_group_dimension_selected == True:
            if isolate_dim_list == None:
                isolate_dim_list = {}
            grouped_dimension_bound = self.transform_grouped_dimension_bound(dimension_bound, group_constraint, isolate_dim_list)
        else:
            grouped_dimension_bound = None
                   
        # Calculating increment budget for optimization
        increment = self.increment_factor(df_grp)
        
        dimScurveList, dimScurveWeights, ScurveElbowDim = self.get_s_curves(dimension_bound, df_grp)

        """optimization process-
            Initialization of minimum bounds or constraint entered by the user for each dimension
            Optimzation on budget and constarints
        """
        oldSpendVec, oldReturn, oldImpVec = self.initial_allocation(dimension_bound)
        if self.use_impression:
            result_df, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, grouped_dimension_bound, dimScurveList, dimScurveWeights, ScurveElbowDim, oldImpVec)
            result_df=result_df[['dimension', 'spend', 'impression', 'return']]
            result_df[['spend', 'impression', 'return']]=result_df[['spend', 'impression', 'return']].round(2)
        else:
            result_df, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, grouped_dimension_bound, dimScurveList, dimScurveWeights, ScurveElbowDim, None)
            result_df=result_df[['dimension', 'spend', 'return']]
            result_df[['spend', 'return']]=result_df[['spend', 'return']].round(2)

        # Calculating other variables for optimization plan for front end
        result_df=self.lift_cal(result_df, budget_per_day, df_spend_dis, days, dimension_bound)
        result_df, summary_metrics_dic, check_discard_json =self.optimizer_result_adjust(discard_json, result_df, df_spend_dis, dimension_bound_actual, budget_per_day, days)
        
        # Df for iterative steps, not displayed in front end
        result_itr_df=result_itr_df.round(2)

        # Optimization confidence score calculation
        optimization_conf_score = self.confidence_score(result_df, df_score_final, df_grp, lst_dim, dimension_bound)

        return result_df, summary_metrics_dic, optimization_conf_score, check_discard_json


class optimizer_iterative_seasonality:
    
    def __init__(self, df_param, constraint_type, target_type, is_weekly_selected):
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
        # Setting if group dimension constraint selected or not as False, updating the flag in execute function if group dimension constraint selected is selected
        self.is_group_dimension_selected = False
        # Precision used for optimization
        self.precision = 1e-0
        # Max iterations used for optimization
        self.max_iter = 500000

    
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
    

    def transform_grouped_dimension_bound(self, dimension_bound, group_constraint, isolate_dim_list):
        """Transforming grouped dimension dictionary to use for constarints for optimization

        Returns:
            Dictionary: grp_dim_bound
        """

        grp_dim_bound = {}

        for dim in group_constraint:
            sub_dim_list = group_constraint[dim]['sub_dimension']
            sub_dim_constraint = group_constraint[dim]['constraints'][1]
            for dim_ in sub_dim_list:
                grp_dim_bound[dim_] = {'dimension' : sub_dim_list,
                                    'constraints':sub_dim_constraint}
                
        for dim_ in isolate_dim_list:
            grp_dim_bound[dim_] = {'dimension' : [dim_],
                                    'constraints':dimension_bound[dim_][1]}
            
        return grp_dim_bound
    
    
    def ini_start_value(self, df_grp, dimension_bound, increment, grouped_dimension_bound):
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

        if self.is_group_dimension_selected == True:
            checked_dim_list = []
            for dim in self.dimension_names:
                if dim in checked_dim_list:
                    continue
                sub_dim_list = grouped_dimension_bound[dim]['dimension']
                groupSpendConstraint = grouped_dimension_bound[dim]['constraints'] 
                agg_iniSpend=sum(oldSpendVec[dim_ini] for dim_ini in sub_dim_list)
                if(agg_iniSpend>=groupSpendConstraint):
                    for sub_dim in sub_dim_list:
                        oldSpendVec[sub_dim] = dimension_bound[sub_dim][0]
                checked_dim_list = checked_dim_list + sub_dim_list
                
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
    

    def get_grouped_dimension_constraint(self, grouped_dimension_bound, dim, newSpendVec):
        """Function to get allocated spend to sub-dimensions under a group and the group dimension's spend constarint
        Returns:
            Spend variable - 
                subDimSpend: Aggregated allocated spend to each dimension
                grp_dim_const: Group dimension spend constraint         
        """
        subDimSpend = 0
        sub_dim_list = grouped_dimension_bound[dim]['dimension']
        grp_dim_const = grouped_dimension_bound[dim]['constraints']
        for sub_dim in sub_dim_list:
            subDimSpend = subDimSpend + newSpendVec[sub_dim]
        return subDimSpend, grp_dim_const
    

    def get_conversion_dimension(self, newSpendVec, dimension_bound, increment, grouped_dimension_bound, newImpVec):
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

            # getting sum of allocated budget to group of dimensions and grouped budget constraint
            if self.is_group_dimension_selected == True:
                subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
            
            # check if spend allocated to a dimension + increment is less or equal to max constarint and get incremental converstions
            if((newSpendVec[dim] + increment)<=dimension_bound[dim][1]):
                # checks if grouped constraints is selected
                if self.is_group_dimension_selected == True:
                    # check if post allocation of increment budget, grouped constraint is satisfied
                    if ((subDimSpend + increment)<=groupSpendConstraint):
                        incBudget[dim] = increment
                    # check if grouped constraint is lies between before and post allocation of increment budget
                    elif((subDimSpend<groupSpendConstraint) & ((subDimSpend + increment)>groupSpendConstraint)):
                        incBudget[dim] = groupSpendConstraint-subDimSpend
                    # if max budget for grouped constraint is reached
                    else:
                        incBudget[dim]=0
                        incReturns[dim]=-1
                        continue
                # if grouped constraints is not selected
                else:
                    incBudget[dim] = increment
       
            # check if spend allocated to a dimension + increment is greater than max constarint and get converstions for remaining budget for that dimension
            elif((newSpendVec[dim]<dimension_bound[dim][1]) & ((newSpendVec[dim] + increment)>dimension_bound[dim][1])):
                # getting remaining increment budget if post increment allocation budget exceeds max bound for a dimension
                temp_incBudget = dimension_bound[dim][1] - newSpendVec[dim]
                # checks if grouped constraints is selected
                if self.is_group_dimension_selected == True:
                    # check if post allocation of increment budget, grouped constraint is satisfied
                    if ((subDimSpend + temp_incBudget)<=groupSpendConstraint):
                        incBudget[dim] = temp_incBudget
                    # check if grouped constraint is lies between before and post allocation of increment budget
                    elif((subDimSpend<groupSpendConstraint) & ((subDimSpend + temp_incBudget)>groupSpendConstraint)):
                        incBudget[dim] = groupSpendConstraint-subDimSpend
                    # if max budget for grouped constraint is reached
                    else:
                        incBudget[dim]=0
                        incReturns[dim]=-1
                        continue
                # if grouped constraints is not selected
                else:
                    incBudget[dim] = temp_incBudget
            
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
    

    def get_s_curves(self, dimension_bound, df_grp):
    
        dimList = list({dim for dim, value in self.d_param.items() if (value['param a']>1.2)})
        
        dimListFiltered_v1 = []
        dimScurveWeights = {}

        for dim in dimList:
            dim_metric = 0
            if self.use_impression:
                dim_metric = (dimension_bound[dim][1]*1000)/dimension_bound[dim][2]
                dimConv = self.s_curve_hill(dim_metric, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                dim_metric = dim_metric/1000
            else:
                dim_metric = dimension_bound[dim][1]
                dimConv = self.s_curve_hill(dim_metric, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
            
            if (dimConv>=1):
                dimListFiltered_v1 = dimListFiltered_v1 + [dim]
                dimScurveWeights[dim] = [dim_metric, dimConv, (dimConv/dim_metric)]
                
        if self.use_impression:
            dimListFiltered_v2 = list({dim for dim in dimListFiltered_v1 if (self.d_param[dim]['param b']>(dimension_bound[dim][0]*1000)/dimension_bound[dim][2])})
        else:
            dimListFiltered_v2 = list({dim for dim in dimListFiltered_v1 if (self.d_param[dim]['param b']>dimension_bound[dim][0])})
        
        dimListFiltered = sorted(dimListFiltered_v2, key=lambda dim: dimScurveWeights[dim][2], reverse=True)

        ScurveElbowDim = {}
        ScurveElbowDim_temp = {}
        for dim in dimListFiltered:
            if self.use_impression:
                df_temp = df_grp[df_grp['dimension']==dim][['impression', 'predictions']].sort_values(by='impression').reset_index(drop=True)
                data = df_temp[df_temp['impression']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            else:
                df_temp = df_grp[df_grp['dimension']==dim][['spend', 'predictions']].sort_values(by='spend').reset_index(drop=True)
                data = df_temp[df_temp['spend']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            rotor = Rotor()
            rotor.fit_rotate(data)
            elbow_index = rotor.get_elbow_index()
            ScurveElbowDim[dim] = data[elbow_index]
            min_bnd = int(ScurveElbowDim[dim][0]*0.75)
            max_bnd = int(ScurveElbowDim[dim][0]*1.25)
            ScurveElbowDim_temp[dim] = data[elbow_index]
            # print(min_bnd, max_bnd+1, int((max_bnd-min_bnd)/50))
            counter = int((max_bnd-min_bnd)/25)
            if counter < 1:
                counter = 1
            for i in range(min_bnd, max_bnd+1, counter):
                target = self.s_curve_hill(i, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])
                if self.use_impression:
                    df_temp = df_temp.append({'impression':i , 'predictions' : target}, ignore_index=True).reset_index(drop=True)
                else:
                    df_temp = df_temp.append({'spend':i , 'predictions' : target}, ignore_index=True).reset_index(drop=True)
            if self.use_impression:
                df_temp = df_temp.sort_values(by='impression').reset_index(drop=True)
                data = df_temp[df_temp['impression']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            else:
                df_temp = df_temp.sort_values(by='spend').reset_index(drop=True)
                data = df_temp[df_temp['spend']<self.d_param[dim]['param b']].reset_index(drop=True).values.tolist()
            rotor = Rotor()
            rotor.fit_rotate(data)
            elbow_index = rotor.get_elbow_index()
            ScurveElbowDim[dim] = data[elbow_index]
        # print('Scurve Elbow before: ',ScurveElbowDim_temp)
        # print('Scurve Elbow after: ',ScurveElbowDim)
        # print('Weights: ',dimScurveWeights)
        
        return dimListFiltered, dimScurveWeights, ScurveElbowDim
     
    
    def adjust_conversions(self, newSpendVec, totalReturn, dimension_bound, budgetGoal, dimScurveList, ScurveElbowDim, dimAdjustConversionPrevious, grouped_dimension_bound, newImpVec):
    
        dimCounter = dimScurveList

        dimScurveAllocationList = []
        dimAdjustConversion = []
        dimNormalSwapList = []
        dimScurveSwapList =[]
        i = 0
        # print("#####Before reinitialization: ",newSpendVec,"#####",totalReturn)
        while (i < len(dimCounter)):
            dimCheck = dimCounter[i]
            i = i + 1
            # print("Mudit", dimCheck)

            dimSpend = newSpendVec[dimCheck]
            dimConversion = totalReturn[dimCheck]
            
            dimCheckSpend = {}
            dimCheckConversion = {}

            if self.is_group_dimension_selected == True:
                subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dimCheck, newSpendVec)              

            if self.use_impression:

                dimImpression = newImpVec[dimCheck]
                dimCheckImpression = {}

                # dimCheckList = list({dim for dim, value in self.d_param.items() if ((newImpVec[dim]>dimImpression) and (newImpVec[dim]>((dimension_bound[dim][0]*1000)/dimension_bound[dimCheck][2])))})
                # dimCheckList = sorted(dimCheckList, key=lambda dim: newImpVec[dim], reverse=True)
                dimCheckList = list({dim for dim, value in self.d_param.items() if ((newSpendVec[dim]>dimSpend) and (newSpendVec[dim]>dimension_bound[dim][0]))})
                dimCheckList = sorted(dimCheckList, key=lambda dim: newSpendVec[dim], reverse=True)

                for dim_idx in dimCheckList:
                    dimSpendItr = newSpendVec[dim_idx]
                    dimImpressionItr = ((dimSpendItr*1000)/(dimension_bound[dimCheck][2]))
                    if ((dimImpressionItr>=((dimension_bound[dimCheck][0]*1000)/dimension_bound[dimCheck][2])) and (dimImpressionItr<=((dimension_bound[dimCheck][1]*1000)/dimension_bound[dimCheck][2]))):
                        tempImp = dimImpressionItr
                    elif (dimImpressionItr>=((dimension_bound[dimCheck][1]*1000)/dimension_bound[dimCheck][2])):
                        tempImp = (dimension_bound[dimCheck][1]*1000)/dimension_bound[dimCheck][2]
                    else:
                        continue

                    tempSpend = (tempImp*dimension_bound[dimCheck][2])/1000

                    if self.is_group_dimension_selected == True:
                        groupSpendAdjust = subDimSpend - dimSpend + tempSpend
                        if (groupSpendAdjust<=groupSpendConstraint):
                            tempSpend = tempSpend
                            tempImp = tempImp
                        elif((subDimSpend<groupSpendConstraint) & (groupSpendAdjust>groupSpendConstraint)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                tempSpend = tempSpend
                                tempImp = tempImp
                            elif((groupSpendConstraint-subDimSpend)>dimSpend):
                                tempSpend = groupSpendConstraint-subDimSpend
                                tempImp = ((tempSpend*1000)/(dimension_bound[dimCheck][2]))
                            else:
                                continue
                        else:
                            continue

                    tempConv = self.s_curve_hill(tempImp, self.d_param[dimCheck]["param a"], self.d_param[dimCheck]["param b"], self.d_param[dimCheck]["param c"])

                    if (tempConv > dimConversion) and (round(tempConv) >= 1):
                        if ((tempConv-dimConversion)/tempConv)>0.01:
                            dimCheckSpend[dim_idx] = tempSpend
                            dimCheckConversion[dim_idx] = tempConv
                            dimCheckImpression[dim_idx] = tempImp
                        else:
                            continue
                    else:
                        continue
                                            
                    adjustSpendSwap = dimSpend + (newSpendVec[dim_idx] - dimCheckSpend[dim_idx])
                    # print(dimSpend," ",newSpendVec[dim_idx]," ",dimCheckSpend[dim_idx])
                    adjustImpressionSwap = (adjustSpendSwap*1000)/dimension_bound[dim_idx][2]
                    adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                    
                    if (adjustSpendSwap<dimension_bound[dim_idx][0]):
                        if (budgetGoal>=(sum(newSpendVec.values())+dimension_bound[dim_idx][0]-adjustSpendSwap)):
                            adjustSpendSwap = dimension_bound[dim_idx][0]
                            adjustImpressionSwap = (adjustSpendSwap*1000)/dimension_bound[dim_idx][2]
                            adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue
                    elif (adjustSpendSwap>dimension_bound[dim_idx][1]):
                        adjustSpendSwap = dimension_bound[dim_idx][1]
                        adjustImpressionSwap = (adjustSpendSwap*1000)/dimension_bound[dim_idx][2]
                        adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])

                    if self.is_group_dimension_selected == True:
                        subDimSpendSwap, groupSpendConstraintSwap = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim_idx, newSpendVec)
                        groupSpendAdjustSwap = subDimSpendSwap - newSpendVec[dim_idx] + adjustSpendSwap
                        if (groupSpendAdjustSwap<=groupSpendConstraintSwap):
                            adjustSpendSwap = adjustSpendSwap
                            adjustImpressionSwap = adjustImpressionSwap
                            adjustReturnSwap = adjustReturnSwap
                        elif((subDimSpendSwap<groupSpendConstraintSwap) & (groupSpendAdjustSwap>groupSpendConstraintSwap)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                adjustSpendSwap = adjustSpendSwap
                                adjustImpressionSwap = adjustImpressionSwap
                                adjustReturnSwap = adjustReturnSwap
                            else:
                                adjustSpendSwap = groupSpendConstraintSwap-subDimSpendSwap
                                adjustImpressionSwap = ((adjustSpendSwap*1000)/(dimension_bound[dim_idx][2]))
                                adjustReturnSwap = self.s_curve_hill(adjustImpressionSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue

                    if (dimConversion+totalReturn[dim_idx])<(adjustReturnSwap+dimCheckConversion[dim_idx]):
                        newSpendVec[dim_idx] = adjustSpendSwap
                        newImpVec[dim_idx] = adjustImpressionSwap
                        totalReturn[dim_idx] = adjustReturnSwap
                        newSpendVec[dimCheck] = dimCheckSpend[dim_idx]
                        newImpVec[dimCheck] = dimCheckImpression[dim_idx]
                        totalReturn[dimCheck] = dimCheckConversion[dim_idx]
                        # print("Swapped ",dim_idx," ",dimCheck)
                        dimNormalSwapList = dimNormalSwapList + [dim_idx]
                        dimScurveSwapList = dimScurveSwapList + [dimCheck]
                        if (dimCheck in dimScurveList) and (newImpVec[dimCheck]>=ScurveElbowDim[dimCheck][0]):
                            dimScurveAllocationList = dimScurveAllocationList + [dimCheck]
                        break

            else:
                dimCheckList = list({dim for dim, value in self.d_param.items() if ((newSpendVec[dim]>dimSpend) and (newSpendVec[dim]>dimension_bound[dim][0]))})
                dimCheckList = sorted(dimCheckList, key=lambda dim: newSpendVec[dim], reverse=True)

                for dim_idx in dimCheckList:
                    dimSpendItr = newSpendVec[dim_idx]
                    if ((dimSpendItr>=dimension_bound[dimCheck][0]) and (dimSpendItr<=dimension_bound[dimCheck][1])):
                        tempSpend = dimSpendItr
                    elif (dimSpendItr>=dimension_bound[dimCheck][1]):
                        tempSpend = dimension_bound[dimCheck][1]
                    else:
                        continue

                    if self.is_group_dimension_selected == True:
                        groupSpendAdjust = subDimSpend - dimSpend + tempSpend
                        if (groupSpendAdjust<=groupSpendConstraint):
                            tempSpend = tempSpend
                        elif((subDimSpend<groupSpendConstraint) & (groupSpendAdjust>groupSpendConstraint)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                tempSpend = tempSpend
                            elif((groupSpendConstraint-subDimSpend)>dimSpend):
                                tempSpend = groupSpendConstraint-subDimSpend
                            else:
                                continue
                        else:
                            continue

                    tempConv = self.s_curve_hill(tempSpend, self.d_param[dimCheck]["param a"], self.d_param[dimCheck]["param b"], self.d_param[dimCheck]["param c"])
                    if (tempConv > dimConversion) and (round(tempConv) >= 1):
                        if ((tempConv-dimConversion)/tempConv)>0.01:
                            dimCheckSpend[dim_idx] = tempSpend
                            dimCheckConversion[dim_idx] = tempConv
                        else:
                            continue
                    else:
                        continue
                    
                    adjustSpendSwap = dimSpend + (newSpendVec[dim_idx] - dimCheckSpend[dim_idx])
                    adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])                        
                    
                    if (adjustSpendSwap<dimension_bound[dim_idx][0]):
                        if (budgetGoal>=(sum(newSpendVec.values())+dimension_bound[dim_idx][0]-dimSpend)):
                            adjustSpendSwap = dimension_bound[dim_idx][0]
                            adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue
                    elif (adjustSpendSwap>dimension_bound[dim_idx][1]):
                            adjustSpendSwap = dimension_bound[dim_idx][1]
                            adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])

                    if self.is_group_dimension_selected == True:
                        subDimSpendSwap, groupSpendConstraintSwap = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim_idx, newSpendVec)
                        groupSpendAdjustSwap = subDimSpendSwap - newSpendVec[dim_idx] + adjustSpendSwap
                        if (groupSpendAdjustSwap<=groupSpendConstraintSwap):
                            adjustSpendSwap = adjustSpendSwap
                            adjustReturnSwap = adjustReturnSwap
                        elif((subDimSpendSwap<groupSpendConstraintSwap) & (groupSpendAdjustSwap>groupSpendConstraintSwap)):
                            if(dim_idx in grouped_dimension_bound[dimCheck]['dimension']):
                                adjustSpendSwap = adjustSpendSwap
                                adjustReturnSwap = adjustReturnSwap
                            else:
                                adjustSpendSwap = groupSpendConstraintSwap-subDimSpendSwap
                                adjustReturnSwap = self.s_curve_hill(adjustSpendSwap, self.d_param[dim_idx]["param a"], self.d_param[dim_idx]["param b"], self.d_param[dim_idx]["param c"])
                        else:
                            continue
                            
                    if (dimConversion+totalReturn[dim_idx])<(adjustReturnSwap+dimCheckConversion[dim_idx]):
                        newSpendVec[dim_idx] = adjustSpendSwap
                        totalReturn[dim_idx] = adjustReturnSwap
                        newSpendVec[dimCheck] = dimCheckSpend[dim_idx]
                        totalReturn[dimCheck] = dimCheckConversion[dim_idx]
                        # print("Swapped ",dim_idx," ",dimCheck)
                        dimNormalSwapList = dimNormalSwapList + [dim_idx]
                        dimScurveSwapList = dimScurveSwapList + [dimCheck]
                        if (dimCheck in dimScurveList) and (newSpendVec[dimCheck]>=ScurveElbowDim[dimCheck][0]):
                            dimScurveAllocationList = dimScurveAllocationList + [dimCheck]
                        break
        
        if dimScurveSwapList:
            dimAdjustConversion = dimScurveSwapList + dimNormalSwapList
            dimMaxAdjust = max(list({value for dim, value in newSpendVec.items() if (dim in dimScurveSwapList)}))
            # print("Value: ",dimMaxAdjust)
            
            # print("Before reinitialization: ",newSpendVec)
            for dim in newSpendVec:
                if dim in dimAdjustConversion :
                    continue
                elif dim in dimAdjustConversionPrevious:
                    continue
                elif newSpendVec[dim]>dimMaxAdjust:
                    continue
                newSpendVec[dim] = dimension_bound[dim][0]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound, dim, newImpVec)
            # print("After reinitialization: ",newSpendVec,"#####",totalReturn)
                
        return newSpendVec, totalReturn, newImpVec, set(dimScurveAllocationList)
    
    
    def allocate_remaining_budget(self, budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, grouped_dimension_bound, iteration, msg, newImpVec):
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

                if self.is_group_dimension_selected == True:
                    subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
                    if (newSpendVec[dim]<dimension_bound_actual[dim][1]) and (subDimSpend<groupSpendConstraint):
                        allocation_dim_list = allocation_dim_list + [dim]
                        newSpendVec_filtered[dim] = newSpendVec[dim]
                else:
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
        
            incrementCalibration = {}
            
            for dim in allocation_dim_list:
                incrementProportion = 0
                budgetRemainDim = 0

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
                    incrementCalibration[dim] = incrementProportion
                else:
                    incrementCalibration[dim] = budgetRemainDim

            if self.is_group_dimension_selected == True:
                for dim in allocation_dim_list:
                    sub_dim_list = grouped_dimension_bound[dim]['dimension']
                    grp_dim_inc_list = list(np.intersect1d(sub_dim_list, list(incrementCalibration.keys())))
                    agg_incSpend=sum(incrementCalibration[dim_inc] for dim_inc in grp_dim_inc_list)
                    subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
                    for grp_dim in grp_dim_inc_list:
                        if(subDimSpend+agg_incSpend>groupSpendConstraint):
                            budgetRemainDimGroup = subDimSpend+agg_incSpend-groupSpendConstraint
                            incrementCalibrationUpdate = incrementCalibration[grp_dim] - ((incrementCalibration[grp_dim]/agg_incSpend)*budgetRemainDimGroup)
                        else:
                            incrementCalibrationUpdate = incrementCalibration[grp_dim]
                        newSpendVec[grp_dim] = newSpendVec[grp_dim] + incrementCalibrationUpdate
                        totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, grp_dim, newImpVec)
                        if(grp_dim!=dim):
                            allocation_dim_list.remove(grp_dim)                
            else:
                for dim in allocation_dim_list:
                    newSpendVec[dim] = newSpendVec[dim] + incrementCalibration[dim]
                    totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)

            iteration+=1
            
            if (iteration>self.max_iter):
                msg = 4002
                # print("#####Iteration: ",iteration)
                raise Exception("Optimal solution not found")

        return newSpendVec, totalReturn, msg, newImpVec
        
        
    def projections_compare(self, newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, grouped_dimension_bound, newImpVec):
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

            if self.is_group_dimension_selected == True:
                subDimSpend, groupSpendConstraint = self.get_grouped_dimension_constraint(grouped_dimension_bound, dim, newSpendVec)              
                subDimSpend_update=subDimSpend-newSpendVec[dim]+dim_spend
                if ((dim_spend>=dimension_bound_actual[dim][0]) and (dim_spend<=dimension_bound_actual[dim][1]) and (subDimSpend_update<=groupSpendConstraint)):
                    if ((round(totalReturn[dim])==round(return_projection[dim])) and (newSpendVec[dim]>dim_spend)):
                        budgetDecrement = budgetDecrement + (newSpendVec[dim] - dim_spend)
                        newSpendVec[dim] = dim_spend
                        totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound_actual, dim, newImpVec)
            else:
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
    
        # add seasonlaity related target when spend is 0
        for dim in self.d_param:
            if (totalReturn[dim]<0):
                newSpendVec[dim]=0
                totalReturn[dim]=0
                if self.use_impression:
                    newImpVec[dim]=0

        return newSpendVec, totalReturn, newImpVec


    def budget_optimize(self, increment_factor, oldSpendVec, oldReturn, budgetGoal, dimension_bound, dimension_bound_actual, grouped_dimension_bound, dimScurveList, dimScurveWeights, ScurveElbowDim, oldImpVec):
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
        check_budget_per = 0.05
        dimScurveCheck = dimScurveList
        dimAdjustConversionPrevious = []
        # print(dimScurveCheck)

        while(budgetGoal > sum(newSpendVec.values())):
            # print(sum(newSpendVec.values()))
            # Get dim with max incremental conversion
            incReturns, incBudget = self.get_conversion_dimension(newSpendVec, dimension_bound, increment, grouped_dimension_bound, newImpVec)  
            dim_idx = max(incReturns, key=incReturns.get)
            # print(iteration," ",dim_idx," ",incReturns[dim_idx]," ",incBudget[dim_idx])
            
            # If incremental conversion is present
            if(incReturns[dim_idx] > 0):
                iteration+=1
                newSpendVec[dim_idx] = newSpendVec[dim_idx] + incBudget[dim_idx]
                totalReturn, newImpVec = self.total_return(newSpendVec, totalReturn, dimension_bound, dim_idx, newImpVec)
                resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)

             # If incremental conversion is not present
            else:
                budgetUntilized_temp = sum(newSpendVec.values())
                if dimScurveCheck:
                    newSpendVec, totalReturn, newImpVec, dimScurveReturn = self.adjust_conversions(newSpendVec, totalReturn, dimension_bound, budgetGoal, dimScurveCheck, ScurveElbowDim, dimAdjustConversionPrevious, grouped_dimension_bound, newImpVec)
                    dimScurveCheck = list(set(dimScurveCheck).difference(dimScurveReturn))
                    dimScurveCheck = sorted(dimScurveCheck, key=lambda dim: dimScurveWeights[dim][2], reverse=True)
                    dimAdjustConversionPrevious = list(set(dimAdjustConversionPrevious + list(dimScurveReturn)))
                    # print(dimScurveReturn)
                    # print("remaining ",dimScurveCheck)
                    # print()
                if((math.isclose(sum(newSpendVec.values()), budgetUntilized_temp, abs_tol=self.precision)) | (sum(newSpendVec.values()) > budgetUntilized_temp)):
                    newSpendVec, totalReturn, msg, newImpVec = self.allocate_remaining_budget(budgetGoal, newSpendVec, dimension_bound_actual, totalReturn, grouped_dimension_bound, iteration, msg, newImpVec)
                    resultIter_df = resultIter_df.append({'spend':sum(newSpendVec.values()) , 'impression': sum(newImpVec.values()), 'return' : sum(totalReturn.values())}, ignore_index=True).reset_index(drop=True)
                    break
                
            # If budget goal is reached, check for dimension with no conversion but some budget is allocated during inital spend allocation
            if(math.isclose(sum(newSpendVec.values()), budgetGoal, abs_tol=self.precision)):
                if (check_noConversion == 0):
                    check_noConversion = 1
                    newSpendVec, totalReturn, newImpVec = self.projections_compare(newSpendVec, totalReturn, dimension_bound_actual, budgetGoal, grouped_dimension_bound, newImpVec)
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

            if(sum(newSpendVec.values())/budgetGoal>=check_budget_per) and (dimScurveCheck):
                # print(sum(newSpendVec.values())/budgetGoal," ",check_budget_per)
                newSpendVec, totalReturn, newImpVec, dimScurveReturn = self.adjust_conversions(newSpendVec, totalReturn, dimension_bound, budgetGoal, dimScurveCheck, ScurveElbowDim, dimAdjustConversionPrevious, grouped_dimension_bound, newImpVec)
                dimScurveCheck = list(set(dimScurveCheck).difference(dimScurveReturn))
                dimScurveCheck = sorted(dimScurveCheck, key=lambda dim: dimScurveWeights[dim][2], reverse=True)
                dimAdjustConversionPrevious = list(set(dimAdjustConversionPrevious + list(dimScurveReturn)))
                # print("#####Iteration: ",iteration)
                # print(dimScurveReturn)
                # print("remaining ",dimScurveCheck)
                # print()
                budget_per = (sum(newSpendVec.values())/budgetGoal)*100
                check_budget_per = (math.ceil(budget_per/5)*5)/100
                if check_budget_per>=0.95:
                    check_budget_per=0.95
            
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

        # print("#####Iteration: ",iteration)
                
        return result_df, resultIter_df, msg
    

    def initial_allocation(self, dimension_bound):
        """Initialization for minimum value entered by the user in the frontend in the dimension level bounds
                Budget: Minimum value entered by the user
                Target, impression (if selected): Respective target, impression based on budget is calculated 
            
        Returns:
            Dictionay - 
                newSpendVec: Budget allocated to each dimension
                totalReturn: Conversion for allocated budget for each dimension
                newImpVec: Impression allocated to each dimension if applicable otherwise null value is allocated
        """
        if self.use_impression:
            oldSpendVec = {dim:value[0] for dim, value in dimension_bound.items()}
            oldImpVec = {dim:((oldSpendVec[dim]*1000)/(dimension_bound[dim][2])) for dim in self.dimension_names}
            oldReturn = {dim:(self.s_curve_hill(oldImpVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])) for dim in self.dimension_names}     
        else:
            oldSpendVec = {dim:value[0] for dim, value in dimension_bound.items()}
            oldReturn = {dim:(self.s_curve_hill(oldSpendVec[dim], self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"])) for dim in self.dimension_names}
            oldImpVec = None
            
        return oldSpendVec, oldReturn, oldImpVec

        
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
        result_df['buget_allocation_new_%'] = (result_df['recommended_budget_for_n_days']/sum(result_df['recommended_budget_for_n_days'])).round(1)
        result_df['estimated_return_%'] = ((result_df['estimated_return_for_n_days']/sum(result_df['estimated_return_for_n_days']))*100).round(1)
        
        result_df=result_df[['dimension', 'recommended_budget_per_day', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_%', 'estimated_return_for_n_days']]
        
        return result_df
    
    
    def optimizer_result_adjust(self, discard_json, df_res, df_spend_dis, dimension_bound, budget_per_day, days, d_weekday, d_month, date_range, freq_type):
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

        df_res['buget_allocation_new_%'] = ((df_res['recommended_budget_for_n_days']/sum(df_res['recommended_budget_for_n_days']))*100).round(1)

        df_res = df_res.merge(df_spend_dis[['dimension', 'median spend', 'mean spend', 'spend']], on='dimension', how='left')
        df_res['total_buget_allocation_old_%'] = ((df_res['spend']/df_res['spend'].sum())*100).round(2)

        if self.constraint_type == 'median':
            df_res['buget_allocation_old_%'] = ((df_res['median spend']/df_res['median spend'].sum())*100)
            df_res['median spend'] = df_res['median spend'].round().astype(int)
            df_res = df_res.rename(columns={"median spend": "original_constraint_budget_per_day"})
        else:
            df_res['buget_allocation_old_%'] = ((df_res['mean spend']/df_res['mean spend'].sum())*100)
            df_res['mean spend'] = df_res['mean spend'].round().astype(int)
            df_res = df_res.rename(columns={"mean spend": "original_constraint_budget_per_day"})

        for dim in self.d_param:
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
        df_res['buget_allocation_old_%']=df_res['buget_allocation_old_%'].round(2)
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
            df_res=df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'median_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%', 'optimized_CPA_ROI', 'current_projections_CPA_ROI']]
        else:
            df_res = df_res.rename(columns={"original_constraint_budget_per_day": "original_mean_budget_per_day"})
            df_res = df_res.rename(columns={"buget_allocation_old_%": "mean_buget_allocation_old_%"})
            df_res=df_res[['dimension', 'original_mean_budget_per_day', 'recommended_budget_per_day', 'total_buget_allocation_old_%', 'mean_buget_allocation_old_%', 'buget_allocation_new_%', 'recommended_budget_for_n_days', 'estimated_return_per_day', 'estimated_return_for_n_days', 'estimated_return_%', 'current_projections_for_n_days', 'current_projections_%', 'optimized_CPA_ROI', 'current_projections_CPA_ROI']]
        
        df_res = df_res.replace({np.nan: None})

        for dim in discard_json:
            df_res.loc[df_res['dimension']==dim, 'current_projections_CPA_ROI'] = None
            df_res.loc[df_res['dimension']==dim, 'optimized_CPA_ROI'] = None

        int_cols = [i for i in df_res.columns if ((i != "dimension") & ('%' not in i) & ('CPA_ROI' not in i))]
        for i in int_cols:
            df_res.loc[df_res[i].values != None, i]=df_res.loc[df_res[i].values != None, i].astype(float).round().astype(int)

        return df_res, summary_metrics_dic,check_discard_json


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


    def execute(self, df_grp, budget, date_range, df_spend_dis, discard_json, dimension_bound, group_constraint, isolate_dim_list, lst_dim, df_score_final):
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

        # Restricting dimensions budget to max conversion budget if enetered budget is greater for any dimension
        dimension_bound_actual = copy.deepcopy(dimension_bound)
        dimension_bound = self.dimension_bound_max_check(dimension_bound)

        days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1
        if self.is_weekly_selected == True:
            days = int(days/7)
            day_name = pd.to_datetime(date_range[0]).day_name()[0:3]
            freq_type = "W-"+day_name
        else:
            freq_type = "D"
        
        # Considering budget per day till 2 decimal points: truncting (and not rounding-off)
        budget_per_day = budget/days
        budget_per_day = (np.trunc(budget_per_day*100)/100)
        # budget_per_day = np.round((budget/days),2)

        # Update if group dimension constraint selected
        if group_constraint!=None:
            self.is_group_dimension_selected = True

        # Transform grouped dimension dict for adding group level constarints/bounds
        if self.is_group_dimension_selected == True:
            if isolate_dim_list == None:
                isolate_dim_list = {}
            grouped_dimension_bound = self.transform_grouped_dimension_bound(dimension_bound, group_constraint, isolate_dim_list)
        else:
            grouped_dimension_bound = None

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

        dimScurveList, dimScurveWeights, ScurveElbowDim = self.get_s_curves(dimension_bound, df_grp)

        """optimization process-
            Initialization of minimum bounds or constraint entered by the user for each dimension
            Optimzation on budget and constarints
            """
        
        oldSpendVec, oldReturn, oldImpVec = self.initial_allocation(dimension_bound)
        if self.use_impression:
            result_df_, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, grouped_dimension_bound, dimScurveList, dimScurveWeights, ScurveElbowDim, oldImpVec)
            result_df_=result_df_[['dimension', 'spend', 'impression', 'return']]
            result_df_[['spend', 'impression', 'return']]=result_df_[['spend', 'impression', 'return']].round()
        else:
            result_df_, result_itr_df, msg = self.budget_optimize(increment, oldSpendVec, oldReturn, budget_per_day, dimension_bound, dimension_bound_actual, grouped_dimension_bound, dimScurveList, dimScurveWeights, ScurveElbowDim, None)
            result_df_=result_df_[['dimension', 'spend', 'return']]
            result_df_[['spend', 'return']]=result_df_[['spend', 'return']].round()

        # print("##### Without Seasonality: ")
        # print(result_df_)

        # Checking distinct combination of daily and monthly for seasonality
        seasonality_combination = []
        for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both", freq=freq_type):
            seasonality_combination = seasonality_combination + [str(day_.weekday())+"_"+str(day_.month)]
        seasonality_combination = set(seasonality_combination)
        # print()
        # print("##### With Seasonality: ")
        # print(seasonality_combination)
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

            result_df_seasonality = self.get_seasonality_result(result_df_, dimension_bound, init_weekday, init_month)
            
            # print(day_month)
            # print(result_df_seasonality)
            # print()

            sol[day_month] = result_df_seasonality.set_index('dimension').T.to_dict('dict')
            # sol_check[day_month] = msg

            init_weekday = [0, 0, 0, 0, 0, 0]
            init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count_day += 1
        # print(sol)
        # for day_month in sol_check.keys():

        #     if sol_check[day_month] != 4001:
        #         raise Exception("Optimal solution not found")

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
        
        # Calculating other variables for optimization plan for front end
        result_df=self.lift_cal(result_df, budget_per_day, df_spend_dis, days, dimension_bound)
        result_df, summary_metrics_dic, check_discard_json = self.optimizer_result_adjust(discard_json, result_df, df_spend_dis, dimension_bound_actual, budget_per_day, days, d_weekday, d_month, date_range, freq_type)
        
        # Df for iterative steps, not displayed in front end
        result_itr_df=result_itr_df.round(2)

        # Optimization confidence score calculation
        optimization_conf_score = self.confidence_score(result_df, df_score_final, df_grp, lst_dim, dimension_bound)

        return result_df, summary_metrics_dic, optimization_conf_score, check_discard_json