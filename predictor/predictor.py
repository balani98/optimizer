import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from .outlier_treatment import outlier_treatment
import warnings
import math
warnings.filterwarnings("ignore")

# Global variable
progress_var = [0, 0]

class predictor:
    def __init__(self, df, date_range, target_type):
        """
        intialization of the class
        df (dataframe) : aggregated dataframe
        date_range (array) : start and end date
        target_type (variable): type of target variable, volume or dollar based on user input
        """

        self.df = df
        self.date_range = date_range

        # check for impressions
        if "impression" in df.columns:
            self.use_impression = True
        else:
            self.use_impression = False
            # target variable, only required when impression is not selected in predictor
            self.target_type = target_type.lower()

        self.df_param = None

        global progress_var
        # progress counter for model fitting for dimensions 
        progress_var = [0, len(self.df.dimension.unique())]

    def s_curve_hill(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))

    def mape(self, actual, pred):
        """this function is not used anymore due to logic update
        calculating mape
        Args:
            actual (series): target
            pred (series): predicted target
        Returns:
            float: mape
        """
        return np.mean(abs(actual - pred) / actual)

    def SMAPE(self, actual, pred):
        """calculating SMAPE
        Args:
            actual (series): target
            pred (series): predicted target
        Returns:
            float: SMAPE
        """
        smape = abs(actual-pred)/(actual + pred)
        smape = np.mean(smape[~smape.isna()])
        return smape

    def RSE(self, actual, pred, nparam):
        """this function is not used anymore due to logic update
        calculating Std. Error of Residual
        Args:
            actual (series): target
            pred (series): predicted target
            nparam: Number of parameters for calculating degrees of freedom
        Returns:
            float: RSE
        """
        RSS = np.sum(np.square(actual - pred))
        rse = math.sqrt(RSS / (len(actual) - nparam))
        return rse
        
    def fit_curve(self, drop_dimension):
        """function to fit the model and find their optimal parameters
        Args:
            drop_dimension (list): dimensions with very less data
        Returns:
            df_param(dataframe) : model parameter
            df_score(dataframe) : smape,corr,drop points
            prediction_df(dataframe) : predictions
            drop_dimension(list) : dimensions with very less data
        """

        # check for impressions, based on it metric/independent variable is decided
        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        params, score = {}, {}

        prediction_df = pd.DataFrame()

        global progress_var

        # initializing progress counter with number of discarded dimensions (if discarded) due to outlier treatment 
        progress_var_drop_dim = len(drop_dimension)
        if (len(drop_dimension)!=0):
            print(drop_dimension)
            progress_var[0] = progress_var_drop_dim
            print("progress_var ", progress_var)

        # model fitting and parameter computation for each dimensions
        for count, dim in enumerate(self.df.dimension.unique()):
            
            # subsetting each dimension
            dim_df = self.df[self.df.dimension == dim].reset_index(drop=True)

            temp_df = pd.DataFrame()

            # min and max bounds for parameter calculation: [min], [max]
            # [shape parameter, inflection parameter, saturation/max parameter]
            bounds = (
                [0.5, dim_df[metric].quantile(0.3), -np.inf],
                [3, dim_df[metric].quantile(1), np.inf],
            )

            # Number of parameters for RSE for degree of freedom: 3(s-curve)
            nparam = 3

            try:
                # model fit and parameter estimation
                popt, pcov = curve_fit(
                    f=self.s_curve_hill,
                    xdata=dim_df[metric],
                    ydata=dim_df["target"],
                    bounds=bounds,
                )
                # prediction based on model fit
                pred = self.s_curve_hill(dim_df[metric], *popt)
                pred = np.where(pred <= 0, 0, pred)
                # storing model parameters along with metric historical median and mean
                params[dim] = list(popt) + [dim_df[metric].median()] + [dim_df[metric].mean()]

                # calculating model statistics: SMAPE and correlation
                # R-square, MAPE, RSE: not used anymore due to logic update
                score[dim] = []
                # score[dim].append(r2_score(dim_df["target"], pred))
                # score[dim].append(self.RSE(dim_df["target"], pred, nparam))
                # score[dim].append(self.mape(dim_df["target"], pred))
                score[dim].append(self.SMAPE(dim_df['target'], pred))
                score[dim].append(pd.Series(pred).corr(dim_df["target"]))
                
                temp_df["date"] = dim_df.date
                temp_df["predictions"] = pred.round(decimals=2)
                temp_df["dimension"] = dim

            except Exception as e:
                print(dim, " exception: ", e)
                # any dimension whose parameters can't be estimated are added to discard dimension list
                drop_dimension.append(dim)

            prediction_df = pd.concat([temp_df, prediction_df], 0)

            progress_var[0] =  progress_var_drop_dim + count + 1

            print("progress_var ", progress_var)

        # post-processing: model statistics and parameters
        df_score = (
            pd.DataFrame(score)
            .T.reset_index()
            .rename(
                columns={"index": "dimension", 0: "SMAPE", 1: "correlation"}
            )
        )

        df_score[["SMAPE", "correlation"]] = df_score[
            ["SMAPE", "correlation"]
        ].round(decimals=2)

        df_param = (
            pd.DataFrame(params)
            .T.reset_index()
            .rename(
                columns={
                    "index": "dimension",
                    0: "param a",
                    1: "param b",
                    2: "param c",
                    3: "median spend",
                    4: "mean spend"
                }
            )
        )

        return df_param, df_score, prediction_df, drop_dimension

    def data_filter(self):
        """filter data:
        (i) where spend and target value both are zero in historical data
        (ii) date range for each dimension selected by the user"""

        self.df = self.df[~((self.df["spend"] == 0) & (self.df["target"] == 0))]

        df_grp_tmp = pd.DataFrame()

        for dim in self.df['dimension'].unique():
            df_grp_ = self.df[self.df['dimension']==dim].reset_index(drop=True)
            df_grp_ = df_grp_[(df_grp_['date']>=self.date_range[dim][0]) & (df_grp_['date']<=self.date_range[dim][1])]
            df_grp_tmp = pd.concat([df_grp_tmp,df_grp_],ignore_index=True)
            
        self.df = df_grp_tmp        

    def predict_dimesion(self, dimension, date_range, budget, cpm=None):
        """this function is not used anymore due to logic update
        function to predict target
         Args:
            dimension (string): dimensions for which predict run
            date_range (array) : start and end date
            budget(int) : budget
        Returns:
            float: target
        """

        days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1

        budget_per_day = budget / days

        param = self.df_param[self.df_param["dimension"] == dimension]

        if self.use_impression:
            x_var = (budget_per_day * 1000) / cpm
        else:
            x_var = budget_per_day

        return days * (
            self.s_curve_hill(
                x_var, param["param a"][0], param["param b"][0], param["param c"][0]
            )
        )
    
    def roi_cpa_outlier_treatment(self, df):
        """function to remove outliers on CPA/ROI (for spend) and conversion rate (for impressions)
        this function is utilized only for plotting graph in the front end
        """
        # check for impression
        if self.use_impression:
            col = 'impression_predictions_rate'
            metric = 'impression'
        else:
            col = 'spend_predictions_rate'
            metric = 'spend'

        df_ = pd.DataFrame()
        
        # check for outlier for each dimension, using z-score method, assigning outliers as -1
        for dim in df['dimension'].unique():
            df_temp = df[df["dimension"] == dim].reset_index(drop=True)
            df_temp_filter=df_temp[df_temp[col]!=-1]
            mean = np.mean(df_temp_filter[col])
            std = np.std(df_temp_filter[col])
            df_temp_filter[col] = np.where((abs((df_temp_filter[col] - mean) / std) > 3), -1, df_temp_filter[col])
            df_temp.drop(columns=[col], inplace=True)
            df_temp = df_temp.merge(df_temp_filter[['date', col]], on='date', how='left').sort_values(by='date').reset_index(drop=True)
            df_ = df_.append(df_temp, ignore_index=True).reset_index(drop=True)
            df_=df_.fillna(-1)

        return df_
    
    def median_mean_datapoints(self):
        """function to calculate median and mean investment/impression based on historic data and respective target for each dimension
        this function is utilized only for plotting these datapoints in the graph in the front end
        """
        median_mean_dic = {}
        for dim in self.df_param['dimension'].unique():
            if self.use_impression:
                median_var = 'impression_median'
                mean_var = 'impression_mean'
            else:
                median_var = 'median spend'
                mean_var = 'mean spend'
            param_ = self.df_param[self.df_param["dimension"] == dim]
            metric_median = self.df_param[self.df_param['dimension']==dim][median_var].values[0]
            metric_mean = self.df_param[self.df_param['dimension']==dim][mean_var].values[0]
            median_mean_dic[dim] = {'median': [round(metric_median, 2), 
                                                round(self.s_curve_hill(
                                                    metric_median, 
                                                    param_["param a"].values[0], 
                                                    param_["param b"].values[0], 
                                                    param_["param c"].values[0]), 2)],
                                    'mean': [round(metric_mean, 2), 
                                                round(self.s_curve_hill(
                                                    metric_mean, 
                                                    param_["param a"].values[0], 
                                                    param_["param b"].values[0], 
                                                    param_["param c"].values[0]), 2)]}
        return median_mean_dic
    
    def dimension_equation(self, a, b, c, dim):
        if self.use_impression:
            metric = " Impression"
        else:
            metric = " Spend"
        X = dim + metric
        eq = str(c) + " * ((" + X + "^" + str(a) + ") / (" + X + "^" + str(a) + " + " + str(b) + "^" + str(a) + "))"
        # return c * ((X ** a) / (X ** a + b ** a))
        return eq
    
    def response_curve_eq(self):
        response_curve_eq = {}
        for dim in self.df_param['dimension'].unique():
            temp_df = self.df_param[self.df_param['dimension']==dim].reset_index()
            response_curve_eq[dim] = self.dimension_equation(round(temp_df['param a'][0],2), round(temp_df['param b'][0],2), round(temp_df['param c'][0],2), dim)
        return response_curve_eq

    def execute(self):
        """execute function
        Note: Formula Used for CPM = (1000 x Spend)/Impression
        Returns:
           df_param(dataframe) : parameter of the model
           df_score_final(dataframe) : quality metric of the curve
           scatter_plot_df(dataframe) : predicted value
           drop_dimension(array) : drop dimension
           df_spend_dis(dataframe) : summary statistics for each dimension
           d_cpm(dictionary) : cpm for each dimension
           median_mean_dic(dictionary) : median and mean historic values for each dimension
        """

        # unique list of dimensions
        dimension_val = list(self.df.dimension.unique())

        # filter dimensions for selected dates and check both spend are target are not zero
        self.data_filter()

        # outlier treatment, using z-score method
        outlier_obj = outlier_treatment(self.df, self.use_impression)
        self.df = outlier_obj.z_outlier()
        
        # calculating metrics for non-discarded dimension, will be utilized in optimize and goal seek
        if self.use_impression:
            df_spend_dis = self.df.groupby('dimension').agg(spend=('spend', 'sum'),
                                                            impression=('impression', 'sum'),
                                                            median_impression=('impression', 'median'),
                                                            mean_impression=('impression', 'mean'),
                                                            return_conv=('target', 'sum')).reset_index()
            df_spend_dis['cpm'] = df_spend_dis["spend"] * 1000 / df_spend_dis["impression"]
            df_spend_dis['median spend'] = (df_spend_dis["median_impression"] * df_spend_dis["cpm"]) / 1000
            df_spend_dis['mean spend'] = (df_spend_dis["mean_impression"] * df_spend_dis["cpm"]) / 1000
            df_spend_dis=df_spend_dis[['dimension', 'spend', 'median spend', 'mean spend', 'return_conv']]
            
            for i in list(set(dimension_val) - set(df_spend_dis['dimension'])):
                df_spend_dis.loc[-1] = [i,0,0,0,0]
                df_spend_dis.index = df_spend_dis.index + 1
        else:
            df_spend_dis = self.df.groupby('dimension').agg(spend=('spend', 'sum'),
                                                            median_spend=('spend', 'median'),
                                                            mean_spend=('spend', 'mean'),
                                                            return_conv=('target', 'sum')).reset_index()
            df_spend_dis.rename({'median_spend': 'median spend', 'mean_spend': 'mean spend'}, axis=1, inplace=True)
            
            for i in list(set(dimension_val) - set(df_spend_dis['dimension'])):
                df_spend_dis.loc[-1] = [i,0,0,0,0]
                df_spend_dis.index = df_spend_dis.index + 1

        # number of data ponts dropped post outlier treatment
        df_drop = outlier_obj.drop_points(self.df)

        # updating discarded dimension list post outlier treatment
        drop_dimension = list(set(dimension_val) - set(self.df.dimension.unique()))

        # model fit and parameter computation
        df_param, df_score, prediction_df, drop_dimension = self.fit_curve(
            drop_dimension
        )

        # post-processing after model fit and parameter computation

        # merging dimension level model statistics and statistics for data points dropped post outlier treatment
        df_score_final = df_score.merge(df_drop, on=["dimension"], how="outer")

        # prediction dataframe for each dimension and data point in historic data
        scatter_plot_df = self.df.merge(
            prediction_df, on=["date", "dimension"], how="inner"
        )
        scatter_plot_df[["spend", "target"]] = scatter_plot_df[
            ["spend", "target"]
        ].round(decimals=2)

        # adding dimension level historic spend contribution to parameter dataframe
        df_contri = (
            scatter_plot_df.groupby("dimension").agg({"spend": "sum"}).reset_index()
        )

        df_contri["spend_%"] = df_contri["spend"] / df_contri["spend"].sum()

        df_param = df_param.merge(
            df_contri[["dimension", "spend_%"]], on="dimension", how="left"
        )

        # discarding dimension and updating dataframes for dimension having prediction as 0 for any spend/impression
        scatter_plot_df_chk = (
            scatter_plot_df.groupby("dimension")
            .agg({"predictions": "sum"})
            .reset_index()
        )
        if (
            len(
                list(
                    scatter_plot_df_chk[scatter_plot_df_chk["predictions"] == 0][
                        "dimension"
                    ]
                )
            )
            > 0
        ):
            for dim in scatter_plot_df_chk[scatter_plot_df_chk["predictions"] == 0][
                "dimension"
            ]:
                drop_dimension.append(dim)
            df_param = df_param[
                ~df_param["dimension"].isin(drop_dimension)
            ].reset_index(drop=True)
        scatter_plot_df = scatter_plot_df[
            ~scatter_plot_df["dimension"].isin(drop_dimension)
        ].reset_index(drop=True)

        self.df_param = df_param

        # calculating CPA/ROI for spend and conversion rate for impression (caculating done based on logic provided by business)
        # cpm dataframe in case of impression and adding to parameter dataframe
        if self.use_impression:
            # formula used: conversion rate = predictions/(impression/1000)
            scatter_plot_df["impression_predictions_rate"] = (scatter_plot_df["predictions"]/(scatter_plot_df["impression"]/1000)).round(decimals=2)
            scatter_plot_df["impression_predictions_rate"] = scatter_plot_df["impression_predictions_rate"].replace([np.inf, -np.inf, np.nan], -1)
            # outlier treatment on conversion rate
            scatter_plot_df = self.roi_cpa_outlier_treatment(scatter_plot_df)
            
            # cpm dataframe calculation
            df_cpm = (
                self.df.groupby("dimension").agg(
                    {"spend": "sum", "impression": [np.sum, np.median, np.mean]}
                )
            ).reset_index()
            df_cpm.columns = ["dimension", "spend", "impression", "impression_median", "impression_mean"]
            df_cpm["cpm"] = df_cpm["spend"] * 1000 / df_cpm["impression"]
            d_cpm = df_cpm[["dimension", "cpm"]].set_index("dimension").to_dict()["cpm"]

            df_param = df_param.merge(
                df_cpm[["dimension", "cpm", "impression_median", "impression_mean"]],
                on=["dimension"],
                how="left"
                )  
        else:
            # formula used:
            #   ROI = predictions/spend
            #   CPA = spend/predictions
            if self.target_type == "revenue":
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["predictions"]/scatter_plot_df["spend"]).round(decimals=2)
            else:
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["spend"]/scatter_plot_df["predictions"]).round(decimals=2)
            scatter_plot_df["spend_predictions_rate"] = scatter_plot_df["spend_predictions_rate"].replace([np.inf, -np.inf, np.nan], -1)
            # outlier treatment on CPA/ROI
            scatter_plot_df = self.roi_cpa_outlier_treatment(scatter_plot_df)

        self.df_param = df_param

        # median and mean historic data points and respective target
        median_mean_dic = self.median_mean_datapoints()
        response_curve_eq_dic = self.response_curve_eq()

        if self.use_impression:
            return df_param, df_score_final, scatter_plot_df, drop_dimension, d_cpm, df_spend_dis, median_mean_dic, response_curve_eq_dic
        else:
            return df_param, df_score_final, scatter_plot_df, drop_dimension, df_spend_dis, median_mean_dic, response_curve_eq_dic


class predictor_with_seasonality:
    def __init__(self, df, target_type, is_weekly_selected):
        """
        intialization of the class
        df (dataframe) : aggregated dataframe
        target_type (variable): type of target variable, volume or dollar based on user input
        is_weekly_selected (variable) : flag if data is at daily or weekly granularity
        """

        self.df = df
        self.is_weekly_selected = is_weekly_selected

        # check for impressions
        if "impression" in df.columns:
            self.use_impression = True
        else:
            self.use_impression = False
            # target variable, only required when impression is not selected in predictor
            self.target_type = target_type.lower()

        self.df_param = None

        global progress_var
        # progress counter for model fitting for dimensions
        progress_var = [0, len(self.df.dimension.unique())]

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
    ):
        """This method performs the scurve function on param X considering daily and monthly seasonalities and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return (
            c * (X[metric] ** a / (X[metric] ** a + b**a))
            + X["weekday_1"] * coeff1
            + X["weekday_2"] * coeff2
            + X["weekday_3"] * coeff3
            + X["weekday_4"] * coeff4
            + X["weekday_5"] * coeff5
            + X["weekday_6"] * coeff6
            + X["month_2"] * coeffm1
            + X["month_3"] * coeffm2
            + X["month_4"] * coeffm3
            + X["month_5"] * coeffm4
            + X["month_6"] * coeffm5
            + X["month_7"] * coeffm6
            + X["month_8"] * coeffm7
            + X["month_9"] * coeffm8
            + X["month_10"] * coeffm9
            + X["month_11"] * coeffm10
            + X["month_12"] * coeffm11
        )

    def s_curve_hill_weekly(
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
    ):
        """This method performs the scurve function on param X considering only daily seasonality and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return (
            c * (X[metric] ** a / (X[metric] ** a + b**a))
            + X["weekday_1"] * coeff1
            + X["weekday_2"] * coeff2
            + X["weekday_3"] * coeff3
            + X["weekday_4"] * coeff4
            + X["weekday_5"] * coeff5
            + X["weekday_6"] * coeff6
        )
    
    def s_curve_hill_monthly(
        self,
        X,
        a,
        b,
        c,
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
    ):
        """This method performs the scurve function on param X considering only monthly seasonality and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return (
            c * (X[metric] ** a / (X[metric] ** a + b**a))
            + X["month_2"] * coeffm1
            + X["month_3"] * coeffm2
            + X["month_4"] * coeffm3
            + X["month_5"] * coeffm4
            + X["month_6"] * coeffm5
            + X["month_7"] * coeffm6
            + X["month_8"] * coeffm7
            + X["month_9"] * coeffm8
            + X["month_10"] * coeffm9
            + X["month_11"] * coeffm10
            + X["month_12"] * coeffm11
        )

    def s_curve_hill_decomp(
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
    ):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y and daily and monthly seasonality component seperately"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return (
            c * (X[metric] ** a / (X[metric] ** a + b**a)),
            
            X["weekday_1"] * coeff1
            + X["weekday_2"] * coeff2
            + X["weekday_3"] * coeff3
            + X["weekday_4"] * coeff4
            + X["weekday_5"] * coeff5
            + X["weekday_6"] * coeff6,
            
            X["month_2"] * coeffm1
            + X["month_3"] * coeffm2
            + X["month_4"] * coeffm3
            + X["month_5"] * coeffm4
            + X["month_6"] * coeffm5
            + X["month_7"] * coeffm6
            + X["month_8"] * coeffm7
            + X["month_9"] * coeffm8
            + X["month_10"] * coeffm9
            + X["month_11"] * coeffm10
            + X["month_12"] * coeffm11
        )

    def s_curve_hill_spend_comp(self, X, a, b, c):
        """This method performs the scurve function on param X without seasonality and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return c * (X[metric] ** a / (X[metric] ** a + b**a))

    def s_curve_hill_spend_imp(self, X, a, b, c):
        """This method performs the scurve function on param X without metric and seasonality and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))

    def mape(self, actual, pred):
        """this function is not used anymore due to logic update
        calculating mape
        Args:
            actual (series): target
            pred (series): predicted target
        Returns:
            float: mape
        """
        return np.mean(abs(actual - pred) / actual)
    
    def RSE(self, actual, pred, nparam):
        """this function is not used anymore due to logic update
        calculating Std. Error of Residual
        Args:
            actual (series): target
            pred (series): predicted target
            nparam: Number of parameters for calculating degrees of freedom
        Returns:
            float: RSE
        """
        RSS = np.sum(np.square(actual - pred))
        rse = math.sqrt(RSS / (len(actual) - nparam))
        return rse

    def SMAPE(self, actual, pred):
        """calculating SMAPE
        Args:
            actual (series): target
            pred (series): predicted target
        Returns:
            float: SMAPE
        """
        smape = abs(actual-pred)/(actual + pred)
        smape = np.mean(smape[~smape.isna()])
        return smape

    def seas_check(self, df_sub):
        """to check weekly and monthly seasonality is applicable for daily and weekly data granularity
        daily granularity: both daily and monthly seasonalities are checked
        weekly granularity: only monthly seasonality is checked
        Note: For seasonality to be applicable (both daily and monthly), 2 cycles of data should be available
        Returns:
            boolean: 0/1
        """
        weekly_seas = 0
        monthly_seas = 0

        # check for daily seasonality
        # this will be checked only for daily data granularity
        if self.is_weekly_selected == False:
            df_count_week = df_sub["weekday"].value_counts()

            if df_count_week[df_count_week >= 2].shape[0] == 7:
                weekly_seas = 1
            else:
                weekly_seas = 0

        # check for monthly seasonality
        df_sub["year"] = df_sub["date"].dt.year

        l_count = []
        for i in df_sub.groupby("year")["month"].unique():
            l_count += list(i)

        df_count_month = pd.Series(l_count).value_counts()

        if df_count_month[df_count_month >= 2].shape[0] == 12:
            monthly_seas = 1
        else:
            monthly_seas = 0

        return weekly_seas, monthly_seas

    def fit_curve(self, drop_dimension):
        """function to fit the model and find their optimal parameters
        Args:
            drop_dimension (list): dimensions with very less data
        Returns:
            df_param(dataframe) : model parameter
            df_score(dataframe) : smape,corr,drop points
            prediction_df(dataframe) : predictions
            drop_dimension(list) : dimensions with very less data
        """

        # check for impressions, based on it metric/independent variable is decided
        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        params, score = {}, {}

        seas_drop = []

        prediction_df = pd.DataFrame()

        global progress_var
        
        # dimension level flags for daily and monthly seasonalities
        weekly_seas_flag = [None] * int(len(self.df.dimension.unique()))
        monthly_seas_flag = [None] * int(len(self.df.dimension.unique()))

        # model fitting and parameter computation for each dimensions
        for count, dim in enumerate(self.df.dimension.unique()):

            # subsetting each dimension
            dim_df = self.df[self.df.dimension == dim].reset_index(drop=True)

            # seasonality check
            weekly_seas, monthly_seas = self.seas_check(dim_df)

            # updating seasonality flag for each dimension
            weekly_seas_flag[count] = weekly_seas
            monthly_seas_flag[count] = monthly_seas

            # dimension discarded if seasonality is not present, rules for discarding dimension:
            # (i) for daily granularity, daily seasonality not present (no need to check for monthly seasonality as first level of seasonality is not present)
            # (ii) for monthly granularity, monthly seasonality not present 
            if (((weekly_seas == 0) and (self.is_weekly_selected == False)) or 
                ((monthly_seas == 0) and (self.is_weekly_selected == True))):
                print(dim)
                seas_drop.append(dim)
                continue

            temp_df = pd.DataFrame()

            # min and max bounds for parameter calculation: [min], [max]
            # [shape parameter, inflection parameter, saturation/max parameter, daily parameters(6), monthly parameters(11)]
            bounds = (
                [
                    0.5,
                    dim_df[metric].quantile(0.3),
                    -np.inf,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    3,
                    dim_df[metric].quantile(1),
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ],
            )

            # Number of parameters for RSE for degree of freedom: 3(s-curve) + 6(weekly) + 11(monthly)
            nparam = 20

            # bounds subset based on daily and weekly granularity
            if self.is_weekly_selected == False:
                # bounds subset if monthly seasonality is not present for daily granularity
                if monthly_seas == 0:
                    bounds = (bounds[0][0:9], bounds[1][0:9])
                    # Number of parameters for RSE for degree of freedom: 3(s-curve) + 6(weekly)
                    nparam = 9
            else:
                bounds = ((bounds[0][0:3]+bounds[0][9:]), (bounds[1][0:3]+bounds[1][9:]))
                # Number of parameters for RSE for degree of freedom: 3(s-curve) + 11(monthly)
                nparam = 14

            try:
                # model fit and parameter estimation, for three cases based on condition:
                # (i) for daily granularity, both daily and monthly seasonalities are present
                # (ii) for daily granularity, only daily seasonality is present
                # (iii) for weekly granularity
                if self.is_weekly_selected == False:
                    if monthly_seas == 1:
                        popt, pcov = curve_fit(
                            f=self.s_curve_hill,
                            xdata=dim_df,
                            ydata=dim_df["target"],
                            bounds=bounds,
                        )
                    else:
                        popt, pcov = curve_fit(
                            f=self.s_curve_hill_weekly,
                            xdata=dim_df,
                            ydata=dim_df["target"],
                            bounds=bounds,
                        )
                        popt = list(popt) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                else:
                    popt, pcov = curve_fit(
                        f=self.s_curve_hill_monthly,
                        xdata=dim_df,
                        ydata=dim_df["target"],
                        bounds=bounds,
                    )
                    popt = list(popt)
                    popt = popt[0:3] + [0, 0, 0, 0, 0, 0] + popt[3:]      

                # any dimension whose overall target prediction is 0 are added to discard dimension list
                if sum(self.s_curve_hill_spend_comp(dim_df, *popt[0:3])) == 0:
                    drop_dimension.append(dim)
                
                # spend/impression predictions, daily seasonality predictions and monthly seasonality predictions seperately
                spend_pred,weekday_pred,monthly_pred = self.s_curve_hill_decomp(dim_df, *popt)
                # prediction based on model fit
                pred = self.s_curve_hill(dim_df, *popt)
                # storing model parameters along with metric historical median and mean
                params[dim] = list(popt) + [dim_df[metric].median()] + [dim_df[metric].mean()]

                # calculating model statistics: SMAPE and correlation
                # R-square, MAPE, RSE: not used anymore due to logic update
                score[dim] = []
                # score[dim].append(r2_score(dim_df["target"], pred))
                # score[dim].append(self.RSE(dim_df["target"], pred, nparam))
                # score[dim].append(self.mape(dim_df["target"], pred))
                score[dim].append(self.SMAPE(dim_df['target'],pred))
                score[dim].append(pd.Series(pred).corr(dim_df["target"]))
                
                temp_df["date"] = dim_df.date
                temp_df['weekday_'] = temp_df['date'].dt.day_name()
                temp_df['month_'] = temp_df['date'].dt.month_name()

                temp_df["spend_prediction"] = spend_pred.round(decimals=2)
                temp_df['weekly_prediction'] = weekday_pred.round(decimals=2)
                temp_df['monthly_prediction'] = monthly_pred.round(decimals=2)
                temp_df["predictions"] = pred.round(decimals=2)

                temp_df["predictions"] = np.where(
                    temp_df["predictions"] > 0, temp_df["predictions"], 0
                )
                temp_df["dimension"] = dim

            except Exception as e:
                print(dim, " exception: ", e)
                # any dimension whose parameters can't be estimated are added to discard dimension list
                drop_dimension.append(dim)

            prediction_df = pd.concat([temp_df, prediction_df], 0)

            progress_var[0] = count + 1

            print("progress_var ", progress_var)

        # raising exception if none of the dimensions have seasonality
        if (((weekly_seas_flag.count(0) == int(len(self.df.dimension.unique()))) and (self.is_weekly_selected == False)) or 
            ((monthly_seas_flag.count(0) == int(len(self.df.dimension.unique()))) and (self.is_weekly_selected == True))):
            raise Exception("Seasonality not available in data")
        
        # post-processing: model statistics and parameters
        df_score = (
            pd.DataFrame(score)
            .T.reset_index()
            .rename(
                columns={"index": "dimension",  0: "SMAPE", 1: "correlation"}
            )
        )

        df_score[["SMAPE", "correlation"]] = df_score[
            ["SMAPE", "correlation"]
        ].round(decimals=2)

        df_param = (
            pd.DataFrame(params)
            .T.reset_index()
            .rename(
                columns={
                    "index": "dimension",
                    0: "param a",
                    1: "param b",
                    2: "param c",
                    3: "weekday 1",
                    4: "weekday 2",
                    5: "weekday 3",
                    6: "weekday 4",
                    7: "weekday 5",
                    8: "weekday 6",
                    9: "month 2",
                    10: "month 3",
                    11: "month 4",
                    12: "month 5",
                    13: "month 6",
                    14: "month 7",
                    15: "month 8",
                    16: "month 9",
                    17: "month 10",
                    18: "month 11",
                    19: "month 12",
                    20: "median spend",
                    21: "mean spend"
                }
            )
        )
        # appending dimensions discarded due model parameters can't be estimated and seasonality not present
        drop_dimension = drop_dimension + seas_drop

        return df_param, df_score, prediction_df, drop_dimension

    def data_filter(self):
        """filter data where spend and target value both are zero in historical data"""

        self.df = self.df[~((self.df["spend"] == 0) & (self.df["target"] == 0))]

    def param_adjust(self, df_param):
        """adjust daily and monthly parameters of seasonality
        Returns:
            df_param(dataframe) : model parameter
        """

        df_param = df_param.set_index("dimension")

        df_self_grp_sum = self.df.groupby("dimension").sum().reset_index()

        df_self_grp_sum.columns = [i.replace("_", " ") for i in df_self_grp_sum.columns]

        for dim in df_param.index:

            df_self_grp_sum_ = df_self_grp_sum[df_self_grp_sum["dimension"] == dim]

            for week in range(1, 7):
                weekday = "weekday " + str(week)

                if df_self_grp_sum_[weekday].values[0] == 0:
                    df_param.loc[dim, weekday] = 0

            for month_ in range(2, 13):
                month = "month " + str(month_)

                if df_self_grp_sum_[month].values[0] == 0:
                    df_param.loc[dim, month] = 0

        return df_param.reset_index()
    
    def roi_cpa_outlier_treatment(self, df):
        """function to remove outliers on CPA/ROI (for spend) and conversion rate (for impressions)
        this function is utilized only for plotting graph in the front end
        """
        # check for impression
        if self.use_impression:
            col = 'impression_predictions_rate'
            metric = 'impression'
        else:
            col = 'spend_predictions_rate'
            metric = 'spend'

        df_ = pd.DataFrame()
        
        # check for outlier for each dimension, using z-score method, assigning outliers as -1
        for dim in df['dimension'].unique():
            df_temp = df[df["dimension"] == dim].reset_index(drop=True)
            df_temp_filter=df_temp[df_temp[col]!=-1]
            mean = np.mean(df_temp_filter[col])
            std = np.std(df_temp_filter[col])
            df_temp_filter[col] = np.where((abs((df_temp_filter[col] - mean) / std) > 3), -1, df_temp_filter[col])
            df_temp.drop(columns=[col], inplace=True)
            df_temp = df_temp.merge(df_temp_filter[['date', col]], on='date', how='left').sort_values(by='date').reset_index(drop=True)
            df_ = df_.append(df_temp, ignore_index=True).reset_index(drop=True)
            df_=df_.fillna(-1)

        return df_
    
    def median_mean_datapoints(self):
        """function to calculate median and mean investment/impression based on historic data and respective target for each dimension
        this function is utilized only for plotting these datapoints in the graph in the front end
        """
        median_mean_dic     = {}
        for dim in self.df_param['dimension'].unique():
            if self.use_impression:
                median_var = 'impression_median'
                mean_var = 'impression_mean'
            else:
                median_var = 'median spend'
                mean_var = 'mean spend'
            param_ = self.df_param[self.df_param["dimension"] == dim]
            # dimension level daily seasonality list
            weekday_ = [0, param_["weekday 1"].values[0],
                        param_["weekday 2"].values[0],
                        param_["weekday 3"].values[0],
                        param_["weekday 4"].values[0],
                        param_["weekday 5"].values[0],
                        param_["weekday 6"].values[0]]
            # dimension level monthly seasonality list
            month_ = [0, param_["month 2"].values[0],
                        param_["month 3"].values[0],
                        param_["month 4"].values[0],
                        param_["month 5"].values[0],
                        param_["month 6"].values[0],
                        param_["month 7"].values[0],
                        param_["month 8"].values[0],
                        param_["month 9"].values[0],
                        param_["month 10"].values[0],
                        param_["month 11"].values[0],
                        param_["month 12"].values[0]]
            metric_median = self.df_param[self.df_param['dimension']==dim][median_var].values[0]
            metric_mean = self.df_param[self.df_param['dimension']==dim][mean_var].values[0]
            median_mean_dic[dim] = {'median': [round(metric_median, 2),
                                                round(self.s_curve_hill_spend_imp(
                                                    metric_median, 
                                                    param_["param a"].values[0], 
                                                    param_["param b"].values[0], 
                                                    param_["param c"].values[0])
                                                    + np.median(weekday_)
                                                    + np.median(month_), 2)],
                                    'mean': [round(metric_mean, 2), 
                                                round(self.s_curve_hill_spend_imp(
                                                    metric_mean, 
                                                    param_["param a"].values[0], 
                                                    param_["param b"].values[0], 
                                                    param_["param c"].values[0])
                                                    + np.mean(weekday_) 
                                                    + np.mean(month_), 2)]}
        return median_mean_dic
    
    def dimension_equation(self, a, b, c, dim, scatter_plot_df):
        
        if self.use_impression:
            metric = " Impression"
        else:
            metric = " Spend"
        X = dim + metric
        metric_eq = str(c) + " * ((" + X + "^" + str(a) + ") / (" + X + "^" + str(a) + " + " + str(b) + "^" + str(a) + "))"
        
        temp_week_df = scatter_plot_df[['dimension', 'weekday_', 'weekly_prediction']].drop_duplicates().reset_index(drop=True)
        weekly_eq = " + ("
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            day_pred = temp_week_df[(temp_week_df['dimension']==dim) & (temp_week_df['weekday_']==day)]['weekly_prediction']
            if (len(day_pred)>0):
                day_pred = round(float(day_pred.values[0]), 2)
            else:
                day_pred = 0.0
            weekly_eq = weekly_eq + day + "_Flag * " + str(day_pred)
            if day != 'Sunday':
                weekly_eq = weekly_eq + " + "
            else:
                weekly_eq = weekly_eq + ")"

        temp_month_df = scatter_plot_df[['dimension', 'month_', 'monthly_prediction']].drop_duplicates().reset_index(drop=True)
        monthly_eq = " + ("
        for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
            month_pred = temp_month_df[(temp_month_df['dimension']==dim) & (temp_month_df['month_']==month)]['monthly_prediction']
            if (len(month_pred)>0):
                month_pred = round(float(month_pred.values[0]), 2)
            else:
                month_pred = 0.0
            monthly_eq = monthly_eq + month + "_Flag * " + str(month_pred)
            if month != 'December':
                monthly_eq = monthly_eq + " + "
            else:
                monthly_eq = monthly_eq + ")"

        eq = metric_eq + weekly_eq + monthly_eq

        return eq
    
    def response_curve_eq(self, scatter_plot_df):
        response_curve_eq = {}
        for dim in self.df_param['dimension'].unique():
            temp_df = self.df_param[self.df_param['dimension']==dim].reset_index()
            response_curve_eq[dim] = self.dimension_equation(round(temp_df['param a'][0],2), round(temp_df['param b'][0],2), round(temp_df['param c'][0],2), dim, scatter_plot_df)
        return response_curve_eq
    
    def execute(self):
        """execute function
        Returns:
           df_param(dataframe) : parameter of the model
           df_score_final(dataframe) : quality metric of the curve
           scatter_plot_df(dataframe) : predicted value
           drop_dimension(array) : drop dimension
           df_spend_dis(dataframe) : summary statistics for each dimension
           d_cpm(dictionary) : cpm for each dimension
           median_mean_dic(dictionary) : median and mean historic values for each dimension
        """

        # unique list of dimensions
        dimension_val = list(self.df.dimension.unique())

        # check both spend are target are not zero
        self.data_filter()

        # outlier treatment, using z-score method
        outlier_obj = outlier_treatment(self.df, self.use_impression)
        self.df = outlier_obj.z_outlier()
        
        # calculating metrics for non-discarded dimension, will be utilized in optimize and goal seek
        if self.use_impression:
            df_spend_dis = self.df.groupby('dimension').agg(spend=('spend', 'sum'),
                                                            impression=('impression', 'sum'),
                                                            median_impression=('impression', 'median'),
                                                            mean_impression=('impression', 'mean'),
                                                            return_conv=('target', 'sum')).reset_index()
            df_spend_dis['cpm'] = df_spend_dis["spend"] * 1000 / df_spend_dis["impression"]
            df_spend_dis['median spend'] = (df_spend_dis["median_impression"] * df_spend_dis["cpm"]) / 1000
            df_spend_dis['mean spend'] = (df_spend_dis["mean_impression"] * df_spend_dis["cpm"]) / 1000
            df_spend_dis=df_spend_dis[['dimension', 'spend', 'median spend', 'mean spend', 'return_conv']]
            
            for i in list(set(dimension_val) - set(df_spend_dis['dimension'])):
        
                df_spend_dis.loc[-1] = [i,0,0,0,0]

                df_spend_dis.index = df_spend_dis.index + 1
        else:
            df_spend_dis = self.df.groupby('dimension').agg(spend=('spend', 'sum'),
                                                            median_spend=('spend', 'median'),
                                                            mean_spend=('spend', 'mean'),
                                                            return_conv=('target', 'sum')).reset_index()
            df_spend_dis.rename({'median_spend': 'median spend', 'mean_spend': 'mean spend'}, axis=1, inplace=True)
            
            for i in list(set(dimension_val) - set(df_spend_dis['dimension'])):
        
                df_spend_dis.loc[-1] = [i,0,0,0,0]

                df_spend_dis.index = df_spend_dis.index + 1

        # number of data ponts dropped post outlier treatment
        df_drop = outlier_obj.drop_points(self.df)
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d')
        self.df["weekday"] = self.df["date"].dt.weekday
        self.df["month"] = self.df["date"].dt.month

        self.df = pd.concat(
            [
                self.df,
                pd.get_dummies(self.df["weekday"], prefix="weekday"),
                pd.get_dummies(self.df["month"], prefix="month"),
            ],
            axis=1,
        )

        col_prsnt = [
            "month_1",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "weekday_0",
            "weekday_1",
            "weekday_2",
            "weekday_3",
            "weekday_4",
            "weekday_5",
            "weekday_6",
        ]

        for col in [p for p in col_prsnt if (p not in self.df.columns)]:
            self.df[col] = 0

        self.df.drop(columns=["month_1", "weekday_0"], inplace=True)

        # updating discarded dimension list post outlier treatment
        drop_dimension = list(set(dimension_val) - set(self.df.dimension.unique()))

        # model fit and parameter computation
        df_param, df_score, prediction_df, drop_dimension = self.fit_curve(
            drop_dimension
        )

        # post-processing after model fit and parameter computation

        # merging dimension level model statistics and statistics for data points dropped post outlier treatment
        df_score_final = df_score.merge(df_drop, on=["dimension"], how="outer")

        # prediction dataframe for each dimension and data point in historic data
        scatter_plot_df = self.df.merge(
            prediction_df, on=["date", "dimension"], how="inner"
        )
        scatter_plot_df[["spend", "target"]] = scatter_plot_df[
            ["spend", "target"]
        ].round(decimals=2)

        # adding dimension level historic spend contribution to parameter dataframe
        df_contri = (
            scatter_plot_df.groupby("dimension").agg({"spend": "sum"}).reset_index()
        )

        df_contri["spend_%"] = df_contri["spend"] / df_contri["spend"].sum()

        df_param = df_param.merge(
            df_contri[["dimension", "spend_%"]], on="dimension", how="left"
        )

        # discarding dimension and updating dataframes for dimension having prediction as 0 for any spend/impression
        scatter_plot_df_chk = (
            scatter_plot_df.groupby("dimension")
            .agg({"predictions": "sum"})
            .reset_index()
        )

        if (
            len(
                list(
                    scatter_plot_df_chk[scatter_plot_df_chk["predictions"] == 0][
                        "dimension"
                    ]
                )
            )
            > 0
        ):

            for dim in scatter_plot_df_chk[scatter_plot_df_chk["predictions"] == 0][
                "dimension"
            ]:
                drop_dimension.append(dim)

            df_param = df_param[
                ~df_param["dimension"].isin(drop_dimension)
            ].reset_index(drop=True)

        scatter_plot_df = scatter_plot_df[
            ~scatter_plot_df["dimension"].isin(drop_dimension)
        ].reset_index(drop=True)

        if self.use_impression == False:
            if self.target_type == "revenue":
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["predictions"]/scatter_plot_df["spend"]).round(decimals=2)
            else:
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["spend"]/scatter_plot_df["predictions"]).round(decimals=2)
            scatter_plot_df["spend_predictions_rate"] = scatter_plot_df["spend_predictions_rate"].replace([np.inf, -np.inf, np.nan], 0)

        self.df_param = df_param

        # calculating CPA/ROI for spend and conversion rate for impression (caculating done based on logic provided by business)
        # cpm dataframe in case of impression and adding to parameter dataframe
        if self.use_impression:
            # formula used: conversion rate = predictions/(impression/1000)
            scatter_plot_df["impression_predictions_rate"] = (scatter_plot_df["predictions"]/(scatter_plot_df["impression"]/1000)).round(decimals=2)
            scatter_plot_df["impression_predictions_rate"] = scatter_plot_df["impression_predictions_rate"].replace([np.inf, -np.inf, np.nan], -1)
            # outlier treatment on conversion rate
            scatter_plot_df = self.roi_cpa_outlier_treatment(scatter_plot_df)
            
            scatter_plot_df = scatter_plot_df[['date', 'spend', 'impression', 'target', 'dimension', 'weekday_',
                            'month_', 'spend_prediction', 'weekly_prediction', 'monthly_prediction',
                            'predictions', 'impression_predictions_rate']]

            # cpm dataframe calculation
            df_cpm = (
                self.df.groupby("dimension").agg(
                    {"spend": "sum", "impression": [np.sum, np.median, np.mean]}
                )
            ).reset_index()
            df_cpm.columns = ["dimension", "spend", "impression", "impression_median", "impression_mean"]
            df_cpm["cpm"] = df_cpm["spend"] * 1000 / df_cpm["impression"]
            d_cpm = df_cpm[["dimension", "cpm"]].set_index("dimension").to_dict()["cpm"]

            df_param = df_param.merge(
                df_cpm[["dimension", "cpm", "impression_median", "impression_mean"]],
                on=["dimension"],
                how="left",
            )

        else:
            # formula used:
            #   ROI = predictions/spend
            #   CPA = spend/predictions
            if self.target_type == "revenue":
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["predictions"]/scatter_plot_df["spend"]).round(decimals=2)
            else:
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["spend"]/scatter_plot_df["predictions"]).round(decimals=2)
            scatter_plot_df["spend_predictions_rate"] = scatter_plot_df["spend_predictions_rate"].replace([np.inf, -np.inf, np.nan], -1)
            # outlier treatment on CPA/ROI
            scatter_plot_df = self.roi_cpa_outlier_treatment(scatter_plot_df)

            scatter_plot_df = scatter_plot_df[['date', 'spend', 'target', 'dimension', 'weekday_',
                            'month_', 'spend_prediction', 'weekly_prediction', 'monthly_prediction',
                            'predictions', 'spend_predictions_rate']]

        # adjust daily and monthly parameters of seasonality
        df_param = self.param_adjust(df_param)
        self.df_param = df_param

        # median and mean historic data points and respective target
        median_mean_dic = self.median_mean_datapoints()
        response_curve_eq_dic = self.response_curve_eq(scatter_plot_df)
                    
        if self.use_impression:
            return (
                df_param,
                df_score_final,
                scatter_plot_df,
                drop_dimension,
                d_cpm,
                df_spend_dis,
                median_mean_dic,
                response_curve_eq_dic
            )
        else:
            return df_param, df_score_final, scatter_plot_df, drop_dimension, df_spend_dis, median_mean_dic, response_curve_eq_dic

# Isolated Functions
def s_curve_hill(X, a, b, c):
    """This method performs the scurve function on param X for non-seasonality and
    Returns the outcome as a varible called y"""
    return c * (X ** a / (X ** a + b ** a))


def predict_dimesion(df_param, dimension, total_days, budget, cpm=None):

    """function to predict target for non-seasonality
     Args:
        df_param (dataframe) : model param dataframe
        dimension (string): dimensions for which predict run
        total_days (int) : number of days
        budget(int) : budget
        cpm (float) : when use impression is true
    Returns:
        float: target
    """

    days = total_days

    budget_per_day = budget / days

    param = df_param[df_param["dimension"] == dimension]

    if "cpm" in df_param.columns:
        x_var = (budget_per_day * 1000) / cpm
    else:
        x_var = budget_per_day

    return days * (
        s_curve_hill(
            x_var,
            param["param a"].values[0],
            param["param b"].values[0],
            param["param c"].values[0],
        )
    )


def s_curve_hill_with_seasonality(
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
    """This method performs the scurve function on param X for seasonality and
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


def predict_dimesion_with_seasonality(
    df_param, dimension, date_range, budget, cpm=None
):

    """function to predict target for seasonality
     Args:
        df_param (dataframe) : model param dataframe
        dimension (string): dimensions for which predict run
        date_range (array) : start and end date
        budget(int) : budget
        cpm (float) : when use impression is true
    Returns:
        float: target
    """

    days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1

    budget_per_day = budget / days

    param = df_param[df_param["dimension"] == dimension]

    init_weekday = [0, 0, 0, 0, 0, 0]
    init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    total_rev = []

    for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both"):

        if "cpm" in df_param.columns:
            x_var = (budget_per_day * 1000) / cpm
        else:
            x_var = budget_per_day

        if day_.weekday() != 0:
            init_weekday[day_.weekday() - 1] = 1

        if day_.month != 1:
            init_month[day_.month - 2] = 1

        rev = s_curve_hill_with_seasonality(
            x_var,
            param["param a"].values[0],
            param["param b"].values[0],
            param["param c"].values[0],
            param["weekday 1"].values[0],
            param["weekday 2"].values[0],
            param["weekday 3"].values[0],
            param["weekday 4"].values[0],
            param["weekday 5"].values[0],
            param["weekday 6"].values[0],
            param["month 2"].values[0],
            param["month 3"].values[0],
            param["month 4"].values[0],
            param["month 5"].values[0],
            param["month 6"].values[0],
            param["month 7"].values[0],
            param["month 8"].values[0],
            param["month 9"].values[0],
            param["month 10"].values[0],
            param["month 11"].values[0],
            param["month 12"].values[0],
            init_weekday,
            init_month,
        )
        if rev < 0:
            rev = 0
        total_rev.append(rev)

        init_weekday = [0, 0, 0, 0, 0, 0]
        init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return int(sum(total_rev))


def progress_bar_var():
    global progress_var
    return progress_var