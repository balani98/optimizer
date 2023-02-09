import pandas as pd
from scipy.optimize import minimize


def dimension_bound(df_param):

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
            dim_bound[dim] = [
                int(
                    (d_param[dim]["impression_median"] * d_param[dim]["cpm"] / 1000)
                    * threshold[0]
                ),
                int(
                    (d_param[dim]["impression_median"] * d_param[dim]["cpm"] / 1000)
                    * threshold[1]
                ),
                d_param[dim]["impression_median"] * d_param[dim]["cpm"] / 1000,
                -100,
                200,
                round(d_param[dim]["cpm"], 2),
             ]
    else:
        for dim in d_param.keys():
            dim_bound[dim] = [
                int(d_param[dim]["median spend"] * threshold[0]),
                int(d_param[dim]["median spend"] * threshold[1]),
                d_param[dim]["median spend"],
                -100,
                200
            ]

    return dim_bound


class optimizer_class:
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
        else:
            self.use_impression = False

    def s_curve_hill(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))

    def objective(self, x, sign=-1.0):

        """defining objective of optimizer

        Returns:
            equation: equation(target) need to be maximize
        """

        count = 0
        y = []

        for count, dim in enumerate(self.d_param):
            y.append(
                self.s_curve_hill(
                    x[count],
                    self.d_param[dim]["param a"],
                    self.d_param[dim]["param b"],
                    self.d_param[dim]["param c"],
                )
            )

        y_final = 0

        for i in range(len(y)):
            y_final = y_final + y[i]

        return sign * y_final

    def spend_constraint(self, x):

        """budget constraint for spend optimizer

        Args:
            x (array): spend array

        Returns:
            equation: constraint
        """

        y_final = 0

        for i in range(len(self.d_param)):
            y_final = y_final + x[i]

        return y_final - self.budget

    def spend_optimizer_init(self, dimension_bound, budget):

        """initialization of optimizer parameter

        Args:
            dimension_bound (dict): dimension bound
            budget(int) : per day budget

        Returns:
            x_init(array): initial start param
            eq_con(equation): constraint equation
            bnds(dictionary): bounds of dimension
        """

        x_init = []
        # for d in self.d_param.keys():
        #     x_init.append(self.d_param[d]['median spend'])

        for d in self.d_param.keys():
            if(dimension_bound[d][0]==0):
                x_init.append((dimension_bound[d][0] + dimension_bound[d][1]) / 2)
            else:
                x_init.append(dimension_bound[d][0])

        self.budget = budget

        eq_con = {"type": "eq", "fun": self.spend_constraint}

        bnds = []
        for dim in self.d_param.keys():
            b = (dimension_bound[dim][0], dimension_bound[dim][1])
            bnds.append(b)

        return x_init, eq_con, bnds

    def impr_constraint(self, x):

        """budget constraint for impression optimizer

        Args:
            x (array): spend array

        Returns:
            equation: constraint
        """

        y_final = 0

        for count, dim in enumerate(self.d_param):
            y_final = y_final + (x[count] * self.dimension_bound[dim][2]) / 1000

        return y_final - self.budget

    def impr_optimizer_init(self, dimension_bound, budget):

        """initialization of optimizer parameter

        Args:
            dimension_bound (dict): dimension bound
            budget(int) : per day budget

        Returns:
            x_init(array): initial start param
            eq_con(equation): constraint equation
            bnds(dictionary): bounds of dimension
        """

        x_init = []
        # for d in self.d_param.keys():
        #     x_init.append(self.d_param[d]['impression_median'])

        for d in self.d_param.keys():
            if(dimension_bound[d][0]==0):
                init_par = (
                dimension_bound[d][0] / dimension_bound[d][2]
                + dimension_bound[d][1] / dimension_bound[d][2]
                ) * (1000 / 2)
                x_init.append(init_par)
            else:
                init_par = (
                dimension_bound[d][0] / dimension_bound[d][2]
                ) * (1000)
                x_init.append(init_par)

        self.budget = budget
        self.dimension_bound = dimension_bound

        eq_con = {"type": "eq", "fun": self.impr_constraint}

        bnds = []
        for dim in self.d_param.keys():
            b = (
                dimension_bound[dim][0] * 1000 / dimension_bound[dim][2],
                dimension_bound[dim][1] * 1000 / dimension_bound[dim][2],
            )
            bnds.append(b)

        return x_init, eq_con, bnds

    def lift_cal(self, sol, budget, days):

        """calculate lift  from optimize spend

        Args:
            sol(optimize object): solution obj
            budget(int) : per day budget
            days(int):time period

        Returns:
            df_res(dataframe): final result
        """

        d_res = {}

        d_res["dimension"] = []

        d_res["recommended_budget_per_day"] = []
        d_res["recommended_budget_for_n_days"] = []

        d_res["curr_spend_per_day"] = []
        d_res["curr_spend_for_n_days"] = []

        d_res["est_opt_target_per_day"] = []
        d_res["est_opt_target_for_n_days"] = []

        d_res["est_curr_target_per_day"] = []
        d_res["est_curr_target_for_n_days"] = []

        for count, dim in enumerate(self.d_param):

            d_res["dimension"].append(dim)
            est_opt_target = self.s_curve_hill(
                sol.x[count],
                self.d_param[dim]["param a"],
                self.d_param[dim]["param b"],
                self.d_param[dim]["param c"],
            )

            est_opt_target=int(round(est_opt_target))

            d_res["est_opt_target_per_day"].append(est_opt_target)
            d_res["est_opt_target_for_n_days"].append(days * est_opt_target)

            d_res["curr_spend_per_day"].append(self.d_param[dim]["spend_%"] * budget)
            d_res["curr_spend_for_n_days"].append(
                days * self.d_param[dim]["spend_%"] * budget
            )

            curr_spend = self.d_param[dim]["spend_%"] * budget

            if self.use_impression:
                d_res["recommended_budget_per_day"].append(
                    round((sol.x[count] * self.dimension_bound[dim][2]) / 1000)
                )
                d_res["recommended_budget_for_n_days"].append(
                    days * round((sol.x[count] * self.dimension_bound[dim][2]) / 1000)
                )
                x_var = curr_spend * 1000 / self.dimension_bound[dim][2]
            else:
                d_res["recommended_budget_per_day"].append(round(sol.x[count]))
                d_res["recommended_budget_for_n_days"].append(days * round(sol.x[count]))
                x_var = curr_spend

            est_curr_target = self.s_curve_hill(
                x_var,
                self.d_param[dim]["param a"],
                self.d_param[dim]["param b"],
                self.d_param[dim]["param c"],
            )
            d_res["est_curr_target_per_day"].append(est_curr_target)
            d_res["est_curr_target_for_n_days"].append(days * est_curr_target)

        df_res = pd.DataFrame(d_res)

        int_cols = [i for i in df_res.columns if i != "dimension"]

        df_res["buget_allocation_new"] = (
            df_res["recommended_budget_per_day"]
            * 100
            / df_res["recommended_budget_per_day"].sum()
        ).round(decimals=2)

        # df_res["buget_allocation_old"] = (
        #     df_res["curr_spend_per_day"] * 100 / df_res["curr_spend_per_day"].sum()
        # ).round(decimals=2)

        df_res["estimated_target_new"] = (
            df_res["est_opt_target_per_day"]
            * 100
            / df_res["est_opt_target_per_day"].sum()
        ).round(decimals=2)

        df_res["estimated_target_old"] = (
            df_res["est_curr_target_per_day"]
            * 100
            / df_res["est_curr_target_per_day"].sum()
        ).round(decimals=2)

        df_day = pd.DataFrame()
        df_day["days"] = [i for i in range(1, days + 1)]
        df_day["conversion_optimize_spend"] = int(
            df_res["est_opt_target_per_day"].sum()
        )
        df_day["conversion_current_spend"] = int(
            df_res["est_curr_target_per_day"].sum()
        )

        df_res[int_cols] = df_res[int_cols].astype(int)

        return df_res, df_day
    
    def optimizer_result_adjust(self, discard_json, df_res, df_spend_dis, budget_per_day, days):
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

                l_col_update = ['dimension','recommended_budget_per_day','recommended_budget_for_n_days','curr_spend_per_day','curr_spend_for_n_days']

                if(col in l_col_update):
                    if(col=='recommended_budget_per_day'):
                        l_append.append(round(discard_json[dim_]))
                    elif(col=='recommended_budget_for_n_days'):
                        l_append.append(round(discard_json[dim_])*days)
                    # elif(col=='curr_spend_per_day'):
                    #     l_append.append(d_dis[dim_])
                    # elif(col=='curr_spend_for_n_days'):
                    #     l_append.append(d_dis[dim_]*days)
                    else:
                        l_append.append(dim_)
                else:
                    l_append.append(None)


            df_res.loc[-1] = l_append
            df_res.index = df_res.index + 1  # shifting index
            df_res = df_res.sort_index()

        df_res['buget_allocation_new'] = df_res['recommended_budget_per_day']/df_res['recommended_budget_per_day'].sum()
        # df_res['buget_allocation_old'] = df_res['curr_spend_per_day']/df_res['curr_spend_per_day'].sum()


        df_res = df_res.merge(df_spend_dis, on='dimension', how='left')
        # df_res['buget_allocation_old'] = df_res['spend']/df_res['spend'].sum()
        # df_res['curr_spend_per_day'] = df_res['buget_allocation_old']*df_res['recommended_budget_per_day'].sum()
        # df_res['curr_spend_for_n_days'] = df_res['curr_spend_per_day']*days

        df_res['buget_allocation_old'] = df_res['median spend']/df_res['median spend'].sum()

        for dim in self.d_param:
            spend_projections = budget_per_day*df_res.loc[df_res['dimension']==dim, 'buget_allocation_old']
            if self.use_impression:
                imp_projections = (spend_projections * 1000)/dimension_bound[dim][2]
                metric_projections = imp_projections
            else:
                metric_projections = spend_projections
            df_res.loc[df_res['dimension']==dim, 'current_projections_per_day'] = self.s_curve_hill(metric_projections, self.d_param[dim]["param a"], self.d_param[dim]["param b"], self.d_param[dim]["param c"]).round(2)
        df_res['current_projections_for_n_days'] = df_res['current_projections_per_day']*days
        df_res['current_projections_allocation'] = ((df_res['current_projections_per_day']/df_res['current_projections_per_day'].sum())*100)

        df_res['buget_allocation_new']=(df_res['buget_allocation_new']*100).round(2)
        df_res['buget_allocation_old']=(df_res['buget_allocation_old']*100).round(2)

        df_res['current_projections_per_day']=df_res['current_projections_per_day'].round().astype(int)
        df_res['current_projections_for_n_days']=df_res['current_projections_for_n_days'].round().astype(int)

#         df_res=pd.merge(df_res, df_spend_dis[['dimension', 'median spend']], on='dimension', how='left')
        df_res['median spend'] = df_res['median spend'].round().astype(int)
        df_res = df_res.rename(columns={"median spend": "original_median_budget_per_day"})

        return df_res


    def budget_range(self, dim_bound):

        budget_min, budget_max = [], []

        for dim in dim_bound.keys():
            budget_min.append(dim_bound[dim][0])
            budget_max.append(dim_bound[dim][1])

        return [sum(budget_min), sum(budget_max)]

    def execute(self, budget, days, dimension_bound, discard_json, df_spend_dis):

        """execute optimizer

        Args:
            budget(int) : budget
            days(int):time period
            dimension_bound(dictionary): bound for each dimension


        Raises:
            Exception: no optimal solution found

        Returns:
            df_res(dataframe): result dataframe
            df_day(dataframe): trend of revenue by day
        """
        budget_per_day = round(budget / days)

        self.d_param = {k: self.d_param[k] for k in dimension_bound.keys()}

        d_param_ = pd.DataFrame(self.d_param)
        d_param_.loc["spend_%", :] = (
            d_param_.loc["spend_%", :] / d_param_.loc["spend_%", :].sum()
        )
        self.d_param = d_param_.to_dict()

        if self.use_impression:
            x_init, eq_con, bnds = self.impr_optimizer_init(
                dimension_bound, budget_per_day
            )
        else:
            x_init, eq_con, bnds = self.spend_optimizer_init(
                dimension_bound, budget_per_day
            )

        sol = minimize(
            self.objective,
            x_init,
            method="SLSQP",
            jac="cs",
            bounds=bnds,
            constraints=[eq_con],
            options={"maxiter": 10000},
        )

        if sol.success == False:
            raise Exception("optimal solution not found")

        df_res, df_day = self.lift_cal(sol, budget_per_day, days)
        
        df_res = self.optimizer_result_adjust(discard_json, df_res, df_spend_dis, budget_per_day, days)
        
#         df_param__ = pd.DataFrame(self.d_param).T.reset_index(drop=False).rename(columns={"index":"dimension"})
#         df_param__['median spend']=df_param__['median spend'].round(2)
#         df_param__=df_param__.rename(columns={"median spend":"original_median_budget_per_day"})
#         df_res=pd.merge(df_res, df_param__[['dimension', 'original_median_budget_per_day']], on='dimension', how='left')
        
        df_res__ = df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'buget_allocation_old', 'buget_allocation_new', 'recommended_budget_for_n_days', 'est_opt_target_per_day', 'est_opt_target_for_n_days', 'estimated_target_new', 'current_projections_for_n_days']]

        return df_res__, df_day
    
    
class optimizer_with_seasonality_class:
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
        else:
            self.use_impression = False

    def s_curve_hill(
        self,
        X,
        a,
        b,
        c,
        coeff1,
        coeff2,
        coeff3,
        coeff4,
        coeff5,
        coeff6,
        coeffm1,
        coeffm2,
        coeffm3,
        coeffm4,
        coeffm5,
        coeffm6,
        coeffm7,
        coeffm8,
        coeffm9,
        coeffm10,
        coeffm11,
        weekday_a,
        month_a,
    ):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return (
            c * (X ** a / (X ** a + b ** a))
            + coeff1 * weekday_a[0]
            + coeff2 * weekday_a[1]
            + coeff3 * weekday_a[2]
            + coeff4 * weekday_a[3]
            + coeff5 * weekday_a[4]
            + coeff6 * weekday_a[5]
            + coeffm1 * month_a[0]
            + coeffm2 * month_a[1]
            + coeffm3 * month_a[2]
            + coeffm4 * month_a[3]
            + coeffm5 * month_a[4]
            + coeffm6 * month_a[5]
            + coeffm7 * month_a[6]
            + coeffm8 * month_a[7]
            + coeffm9 * month_a[8]
            + coeffm10 * month_a[9]
            + coeffm11 * month_a[10]
        )

    def s_curve_hill_(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))

    def objective(self, x, sign=-1.0):

        """defining objective of optimizer

        Returns:
            equation: equation(target) need to be maximize
        """

        count = 0
        y = []

        for count, dim in enumerate(self.d_param):
            y.append(
                self.s_curve_hill_(
                    x[count],
                    self.d_param[dim]["param a"],
                    self.d_param[dim]["param b"],
                    self.d_param[dim]["param c"],
                )
            )

        y_final = 0

        for i in range(len(y)):
            y_final = y_final + y[i]

        return sign * y_final

    def spend_constraint(self, x):

        """budget constraint for spend optimizer

        Args:
            x (array): spend array

        Returns:
            equation: constraint
        """

        y_final = 0

        for i in range(len(self.d_param)):
            y_final = y_final + x[i]

        return y_final - self.budget

    def spend_optimizer_init(self, dimension_bound, budget):

        """initialization of optimizer parameter

        Args:
            dimension_bound (dict): dimension bound
            budget(int) : per day budget

        Returns:
            x_init(array): initial start param
            eq_con(equation): constraint equation
            bnds(dictionary): bounds of dimension
        """

        x_init = []
        # for d in self.d_param.keys():
        #     x_init.append(self.d_param[d]['median spend'])

        for d in self.d_param.keys():
            if(dimension_bound[d][0]==0):
                x_init.append((dimension_bound[d][0] + dimension_bound[d][1]) / 2)
            else:
                x_init.append(dimension_bound[d][0])

        self.budget = budget

        eq_con = {"type": "eq", "fun": self.spend_constraint}

        bnds = []
        for dim in self.d_param.keys():
            b = (dimension_bound[dim][0], dimension_bound[dim][1])
            bnds.append(b)

        return x_init, eq_con, bnds

    def impr_constraint(self, x):

        """budget constraint for impression optimizer

        Args:
            x (array): spend array

        Returns:
            equation: constraint
        """

        y_final = 0

        for count, dim in enumerate(self.d_param):
            y_final = y_final + (x[count] * self.dimension_bound[dim][2]) / 1000

        return y_final - self.budget

    def impr_optimizer_init(self, dimension_bound, budget):

        """initialization of optimizer parameter

        Args:
            dimension_bound (dict): dimension bound
            budget(int) : per day budget

        Returns:
            x_init(array): initial start param
            eq_con(equation): constraint equation
            bnds(dictionary): bounds of dimension
        """

        x_init = []
        # for d in self.d_param.keys():
        #     x_init.append(self.d_param[d]['impression_median'])

        for d in self.d_param.keys():
            if(dimension_bound[d][0]==0):
                init_par = (
                dimension_bound[d][0] / dimension_bound[d][2]
                + dimension_bound[d][1] / dimension_bound[d][2]
                ) * (1000 / 2)
                x_init.append(init_par)
            else:
                init_par = (
                dimension_bound[d][0] / dimension_bound[d][2]
                ) * (1000)
                x_init.append(init_par)

        self.budget = budget
        self.dimension_bound = dimension_bound

        eq_con = {"type": "eq", "fun": self.impr_constraint}

        bnds = []
        for dim in self.d_param.keys():
            b = (
                dimension_bound[dim][0] * 1000 / dimension_bound[dim][2],
                dimension_bound[dim][1] * 1000 / dimension_bound[dim][2],
            )
            bnds.append(b)

        return x_init, eq_con, bnds

    def lift_cal(self, sol, budget, d_weekday, d_month):

        """calculate lift  from optimize spend

        Args:
            sol(optimize object): solution obj
            budget(int) : per day budget
            days(int):time period

        Returns:
            df_res(dataframe): final result
        """

        df_cal = pd.DataFrame()

        for count_, day in enumerate(sol):

            d_res = {}

            if count_ == 0:
                d_res["dimension"] = []

            d_res["est_opt_target_" + str(day) + "_day"] = []
            d_res["curr_spend_" + str(day) + "_day"] = []
            d_res["recommended_budget_" + str(day) + "_day"] = []
            d_res["est_curr_target_" + str(day) + "_day"] = []

            for count, dim in enumerate(self.d_param):

                if count_ == 0:
                    d_res["dimension"].append(dim)

                est_opt_target = self.s_curve_hill(
                    sol[day].x[count],
                    self.d_param[dim]["param a"],
                    self.d_param[dim]["param b"],
                    self.d_param[dim]["param c"],
                    self.d_param[dim]["weekday 1"],
                    self.d_param[dim]["weekday 2"],
                    self.d_param[dim]["weekday 3"],
                    self.d_param[dim]["weekday 4"],
                    self.d_param[dim]["weekday 5"],
                    self.d_param[dim]["weekday 6"],
                    self.d_param[dim]["month 2"],
                    self.d_param[dim]["month 3"],
                    self.d_param[dim]["month 4"],
                    self.d_param[dim]["month 5"],
                    self.d_param[dim]["month 6"],
                    self.d_param[dim]["month 7"],
                    self.d_param[dim]["month 8"],
                    self.d_param[dim]["month 9"],
                    self.d_param[dim]["month 10"],
                    self.d_param[dim]["month 11"],
                    self.d_param[dim]["month 12"],
                    d_weekday[day],
                    d_month[day],
                )

                if est_opt_target < 0:
                    est_opt_target = 0

                est_opt_target = int(round(est_opt_target))
                d_res["est_opt_target_" + str(day) + "_day"].append(est_opt_target)

                d_res["curr_spend_" + str(day) + "_day"].append(
                    self.d_param[dim]["spend_%"] * budget
                )

                curr_spend = self.d_param[dim]["spend_%"] * budget

                if self.use_impression:
                    d_res["recommended_budget_" + str(day) + "_day"].append(
                        (sol[day].x[count] * self.dimension_bound[dim][2]) / 1000
                    )

                    x_var = curr_spend * 1000 / self.dimension_bound[dim][2]
                else:
                    d_res["recommended_budget_" + str(day) + "_day"].append(
                        sol[day].x[count]
                    )

                    x_var = curr_spend

                est_curr_target = self.s_curve_hill(
                    x_var,
                    self.d_param[dim]["param a"],
                    self.d_param[dim]["param b"],
                    self.d_param[dim]["param c"],
                    self.d_param[dim]["weekday 1"],
                    self.d_param[dim]["weekday 2"],
                    self.d_param[dim]["weekday 3"],
                    self.d_param[dim]["weekday 4"],
                    self.d_param[dim]["weekday 5"],
                    self.d_param[dim]["weekday 6"],
                    self.d_param[dim]["month 2"],
                    self.d_param[dim]["month 3"],
                    self.d_param[dim]["month 4"],
                    self.d_param[dim]["month 5"],
                    self.d_param[dim]["month 6"],
                    self.d_param[dim]["month 7"],
                    self.d_param[dim]["month 8"],
                    self.d_param[dim]["month 9"],
                    self.d_param[dim]["month 10"],
                    self.d_param[dim]["month 11"],
                    self.d_param[dim]["month 12"],
                    d_weekday[day],
                    d_month[day],
                )

                if est_curr_target < 0:
                    est_curr_target = 0

                d_res["est_curr_target_" + str(day) + "_day"].append(est_curr_target)

            df_res = pd.DataFrame(d_res)

            df_cal = pd.concat([df_cal, df_res], axis=1)

        df_cal["est_opt_target_for_n_days"] = df_cal[
            [col for col in df_cal.columns if (col.startswith("est_opt_"))]
        ].sum(1)

        df_cal["recommended_budget_for_n_days"] = df_cal[
            [col for col in df_cal.columns if (col.startswith("recommended_budget_"))]
        ].sum(1)

        df_cal["est_curr_target_for_n_days"] = df_cal[
            [col for col in df_cal.columns if (col.startswith("est_curr_"))]
        ].sum(1)

        df_cal["curr_spend_for_n_days"] = df_cal[
            [col for col in df_cal.columns if (col.startswith("curr_spend_"))]
        ].sum(1)

        int_cols = [i for i in df_cal.columns if i != "dimension"]

        df_cal["buget_allocation_new"] = (
            (
                df_cal["recommended_budget_for_n_days"]
                / df_cal["recommended_budget_for_n_days"].sum()
            )
            * 100
        ).round(decimals=2)

        # df_cal["buget_allocation_old"] = (
        #     (df_cal["curr_spend_for_n_days"] / df_cal["curr_spend_for_n_days"].sum())
        #     * 100
        # ).round(decimals=2)

        df_cal["estimated_target_new"] = (
            (
                df_cal["est_opt_target_for_n_days"]
                / df_cal["est_opt_target_for_n_days"].sum()
            )
            * 100
        ).round(decimals=2)

        df_cal["estimated_target_old"] = (
            (
                pd.Series(df_cal["est_curr_target_for_n_days"])
                / df_cal["est_curr_target_for_n_days"].sum()
            )
            * 100
        ).round(decimals=2)

        df_oc = (
            df_cal[[col for col in df_cal.columns if (col.startswith("est_opt_"))]]
            .sum(0)
            .reset_index()
            .rename(columns={0: "conversion_optimize_spend"})
        )
        df_oc = df_oc[df_oc["index"] != "est_opt_target_for_n_days"]

        df_cc = (
            df_cal[[col for col in df_cal.columns if (col.startswith("est_curr_"))]]
            .sum(0)
            .reset_index()
            .rename(columns={0: "conversion_current_spend"})
        )
        df_cc = df_cc[df_cc["index"] != "est_curr_target_for_n_days"]

        df_day = pd.concat([df_cc, df_oc], ignore_index=True, axis=1).rename(
            columns={1: "conversion_current_spend", 3: "conversion_optimize_spend"}
        )

        day_a = []

        for day_ in sol.keys():
            day_a.append(day_)

        df_day["days"] = day_a

        df_cal["curr_spend_per_day"] = (
            df_cal["curr_spend_for_n_days"] / len(sol.keys())
        ).astype(int)
        df_cal["est_curr_target_per_day"] = (
            df_cal["est_curr_target_for_n_days"] / len(sol.keys())
        ).astype(int)
        df_cal["recommended_budget_for_n_days"] = (
            df_cal["recommended_budget_for_n_days"]
        ).round().astype(int)
        df_cal["recommended_budget_per_day"] = (
            df_cal["recommended_budget_for_n_days"] / len(sol.keys())
        ).round().astype(int)
        df_cal["est_opt_target_per_day"] = (
            df_cal["est_opt_target_for_n_days"] / len(sol.keys())
        ).round().astype(int)

        df_cal[int_cols] = df_cal[int_cols].astype(int)

        return (
            df_cal,
            df_day[["conversion_current_spend", "conversion_optimize_spend", "days"]],
        )

    
    def optimizer_result_adjust_seasonality(self,discard_json,df_res,df_spend_dis,budget_per_day,dimension_bound,d_weekday,d_month,date_range):
        """re-calculation of result based on discarded dimension budget

        Args:
            discard_json (json): key: discarded dimension ,value: spend
            df_res (dataframe): res dataframe from optimizer
            df_spend_dis (dataframe): spend distribution
            date_range (list): [start_date,end_date]

        Returns:
            dataframe: recal of optimizer result for discarded dimension
        """

        days = ( pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0]) ).days + 1

        discard_json = {chnl:discard_json[chnl] for chnl in discard_json.keys() if(discard_json[chnl]!=0)}

        d_dis = df_spend_dis.set_index('dimension').to_dict()['spend']

        for dim_ in discard_json.keys():
            l_append = []
            for col in df_res.columns:

                l_col_update = ['dimension','recommended_budget_per_day','recommended_budget_for_n_days','curr_spend_per_day','curr_spend_for_n_days']

                if(col in l_col_update):
                    if(col=='recommended_budget_per_day'):
                        l_append.append(round(discard_json[dim_]))
                    elif(col=='recommended_budget_for_n_days'):
                        l_append.append(round(discard_json[dim_])*days)
                    # elif(col=='curr_spend_per_day'):
                    #     l_append.append(d_dis[dim_])
                    # elif(col=='curr_spend_for_n_days'):
                    #     l_append.append(d_dis[dim_]*days)
                    else:
                        l_append.append(dim_)
                else:
                    l_append.append(None)


            df_res.loc[-1] = l_append
            df_res.index = df_res.index + 1  # shifting index
            df_res = df_res.sort_index()

        df_res['buget_allocation_new'] = df_res['recommended_budget_per_day']/df_res['recommended_budget_per_day'].sum()
        # df_res['buget_allocation_old'] = df_res['curr_spend_per_day']/df_res['curr_spend_per_day'].sum()

        df_res = df_res.merge(df_spend_dis,on='dimension',how='left')
        # df_res['buget_allocation_old'] = df_res['spend']/df_res['spend'].sum()
        # df_res['curr_spend_per_day'] = df_res['buget_allocation_old']*df_res['recommended_budget_per_day'].sum()
        # df_res['curr_spend_for_n_days'] = df_res['curr_spend_per_day']*days
        
        df_res['buget_allocation_old'] = df_res['median spend']/df_res['median spend'].sum()

        for dim in self.d_param:
            spend_projections = budget_per_day*df_res.loc[df_res['dimension']==dim, 'buget_allocation_old']
            if self.use_impression:
                imp_projections = (spend_projections * 1000)/dimension_bound[dim][2]
                metric_projections = imp_projections
            else:
                metric_projections = spend_projections
            target_projection = 0
            for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both"):
                target_projection = target_projection + self.s_curve_hill(metric_projections,
                                                            self.d_param[dim]["param a"],
                                                            self.d_param[dim]["param b"],
                                                            self.d_param[dim]["param c"],
                                                            self.d_param[dim]["weekday 1"],
                                                            self.d_param[dim]["weekday 2"],
                                                            self.d_param[dim]["weekday 3"],
                                                            self.d_param[dim]["weekday 4"],
                                                            self.d_param[dim]["weekday 5"],
                                                            self.d_param[dim]["weekday 6"],
                                                            self.d_param[dim]["month 2"],
                                                            self.d_param[dim]["month 3"],
                                                            self.d_param[dim]["month 4"],
                                                            self.d_param[dim]["month 5"],
                                                            self.d_param[dim]["month 6"],
                                                            self.d_param[dim]["month 7"],
                                                            self.d_param[dim]["month 8"],
                                                            self.d_param[dim]["month 9"],
                                                            self.d_param[dim]["month 10"],
                                                            self.d_param[dim]["month 11"],
                                                            self.d_param[dim]["month 12"],
                                                            d_weekday[str(day_.date())],
                                                            d_month[str(day_.date())])
            df_res.loc[df_res['dimension']==dim, 'current_projections_for_n_days'] = target_projection
        df_res['current_projections_for_n_days'] = df_res['current_projections_for_n_days'].round().astype(int)
        df_res['current_projections_per_day'] = (df_res['current_projections_for_n_days']/days).round().astype(int)
        df_res['current_projections_allocation'] = ((df_res['current_projections_for_n_days']/df_res['current_projections_for_n_days'].sum())*100).round(2)

        df_res['buget_allocation_new']=(df_res['buget_allocation_new']*100).round(2)
        df_res['buget_allocation_old']=(df_res['buget_allocation_old']*100).round(2)

#         df_res=pd.merge(df_res, df_spend_dis[['dimension', 'median spend']], on='dimension', how='left')
        df_res['median spend']=df_res['median spend'].round().astype(int)
        df_res=df_res.rename(columns={"median spend":"original_median_budget_per_day"})

        return df_res   


    def budget_range(self, dim_bound):

        budget_min, budget_max = [], []

        for dim in dim_bound.keys():
            budget_min.append(dim_bound[dim][0])
            budget_max.append(dim_bound[dim][1])

        return [sum(budget_min), sum(budget_max)]

    def execute(self, budget, date_range, dimension_bound, discard_json, df_spend_dis):

        """execute optimizer

        Args:
            budget(int) : budget
            date_range(array):[start day,end day] (mm-dd-YYYY)
            dimension_bound(dictionary): bound for each dimension


        Raises:
            Exception: no optimal solution found

        Returns:
            df_res(dataframe): result dataframe
            df_day(dataframe): trend of revenue by day
        """

        # days = (
        #     int(
        #         pd.to_datetime(date_range[1]).dayofyear
        #         - pd.to_datetime(date_range[0]).dayofyear
        #     )
        #     + 1
        # )

        days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1

        budget_per_day = round(budget / days)

        self.d_param = {k: self.d_param[k] for k in dimension_bound.keys()}

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

        for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both"):

            if day_.weekday() != 0:
                init_weekday[day_.weekday() - 1] = 1

            if day_.month != 1:
                init_month[day_.month - 2] = 1

            d_weekday[str(day_.date())] = init_weekday
            d_month[str(day_.date())] = init_month

            if self.use_impression:
                x_init, eq_con, bnds = self.impr_optimizer_init(
                    dimension_bound, budget_per_day
                )
            else:
                x_init, eq_con, bnds = self.spend_optimizer_init(
                    dimension_bound, budget_per_day
                )

            self.weekday_tmp = d_weekday[str(day_.date())]
            self.month_tmp = d_month[str(day_.date())]

            sol_ = minimize(
                self.objective,
                x_init,
                method="SLSQP",
                jac="cs",
                bounds=bnds,
                constraints=[eq_con],
                options={"maxiter": 10000},
            )

            sol[str(day_.date())] = sol_

            init_weekday = [0, 0, 0, 0, 0, 0]
            init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count_day += 1

        for day_ in sol.keys():

            if sol[day_].success == False:
                raise Exception("optimal solution not found")

        df_res, df_day = self.lift_cal(sol, budget_per_day, d_weekday, d_month)
        
        df_res = self.optimizer_result_adjust_seasonality(discard_json,df_res,df_spend_dis,budget_per_day,dimension_bound,d_weekday,d_month,date_range)

#         df_param__ = pd.DataFrame(self.d_param).T.reset_index(drop=False).rename(columns={"index":"dimension"})
#         df_param__['median spend']=df_param__['median spend'].round(2)
#         df_param__=df_param__.rename(columns={"median spend":"original_median_budget_per_day"})
#         df_res=pd.merge(df_res, df_param__[['dimension', 'original_median_budget_per_day']], on='dimension', how='left')
        
        df_res__ = df_res[['dimension', 'original_median_budget_per_day', 'recommended_budget_per_day', 'buget_allocation_old', 'buget_allocation_new', 'recommended_budget_for_n_days', 'est_opt_target_per_day', 'est_opt_target_for_n_days', 'estimated_target_new', 'current_projections_for_n_days']]

        return df_res__, df_day