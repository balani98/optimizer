from itertools import dropwhile
import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_numeric_dtype, is_float_dtype
import datetime


class explorer:
    def __init__(self, df):
        """
        intialization of the class
        """
        self.df = df
        self.dimension = None
        self.date = None
        self.spend = None
        self.target = None
        self.cpm = None
        self.use_impression = False
        self.is_weekly_selected = False
        self.convert_to_weekly = False

    def numeric_check(self, numeric_):
        """perform data validation for numeric columns

        Args:
            numeric_ (_type_): spend/target/cpm
        Raises:
            Exception 5002: value error
            Exception 5003: type error
        """

        row_count = self.df.shape[0]

        null_val = self.df[getattr(self, numeric_)].isnull().sum()

        if null_val == row_count:
            raise Exception(5002)

        if not (is_numeric_dtype(self.df[getattr(self, numeric_)])) | (
            is_float_dtype(self.df[getattr(self, numeric_)])
        ):
            raise Exception(5003)

    def date_check(self):
        """perform data audit for date format, null values

        Raises:
            Exception 5002: value error
            Exception 5004: date format error
        """
        _format = "%Y-%m-%d"
        # _format = "%d-%b-%y"
        # _format = "%m-%d-%y"

        null_val = self.df[self.date].isnull().sum()

        if not null_val == 0:
            raise Exception(5002)

        try:
            self.df[self.date] = pd.to_datetime(self.df[self.date], format=_format)
        except:
            raise Exception(5004)

    def dimension_check(self):

        """checking for null in dimension

        Raises:
            Exception 5002: value error
            Exception 5003: type error
        """

        null_val = self.df[self.dimension].isna().any(1).sum()

        row_count = self.df.shape[0]

        if null_val == row_count:
            raise Exception(5002)

        for dim in self.dimension:

            if (is_numeric_dtype(self.df[dim])) | (is_float_dtype(self.df[dim])):
                raise Exception(5003)

    def data_aggregation(self):

        """data aggregation at selected dimension,day level

        Returns:
            dataframe: aggregated dataframe
        """
        _format = "%Y-%m-%d"
        if pd.to_datetime(self.df[self.date], format=_format, errors='coerce').notnull().all(): 
            self.df[self.date] = pd.to_datetime(self.df[self.date], format=_format)
        self.df = self.df.rename(
            columns={self.spend: "spend", self.target: "target", self.date: "date"}
        )

        if self.use_impression == False:

            group_l = self.dimension + ["date"]

            df_grp = (
                self.df.groupby(group_l)
                .agg({"spend": "sum", "target": "sum"})
                .reset_index()
            )

        else:

            self.df = self.df.rename(columns={self.cpm: "cpm"})

            group_l = self.dimension + ["date"]

            self.df["impression"] = (self.df["spend"] * 1000) / self.df["cpm"]
            # self.df["impression"] = self.df["impression"].fillna(0)

            df_grp = (
                self.df.groupby(group_l)
                .agg({"spend": "sum", "impression": "sum", "target": "sum"})
                .reset_index()
            )

            # df_cpm_mean = self.df.groupby(self.dimension).agg({"cpm": "mean"})

        df_grp["_dimension_"] = ""

        count = 1

        if len(self.dimension) > 1:
            for dim in self.dimension:

                if count != len(self.dimension):
                    df_grp["_dimension_"] = df_grp["_dimension_"] + df_grp[dim] + "_"
                else:
                    df_grp["_dimension_"] = df_grp["_dimension_"] + "_" + df_grp[dim]

                count += 1
        else:
            df_grp["_dimension_"] = df_grp[self.dimension[0]]

        df_grp.drop(columns=self.dimension, inplace=True)

        df_grp.rename(columns={"_dimension_": "dimension"}, inplace=True)

        if self.convert_to_weekly == True:
            df_grp = self.convert_to_weekly_granularity(df_grp)

        df_grp = self.impute_missing_date(df_grp)

        if self.use_impression:
            return df_grp
        else:
            return df_grp

    def convert_to_weekly_granularity(self, df_grp):
        
        """convert daily data granularity to weekly data granularity - Week Starting Monday

        Returns:
            dataframe: dataframe with weekly data
        """
        self.is_weekly_selected = True
        
        # Aggregating to weekly level
        if self.use_impression == False:
            df_grp = df_grp.groupby(['dimension', pd.Grouper(key='date', freq='W-SUN')])['spend', 'target'].sum().reset_index().sort_values(['dimension', 'date'])
        else:
            df_grp = df_grp.groupby(['dimension', pd.Grouper(key='date', freq='W-SUN')])['spend', 'impression', 'target'].sum().reset_index().sort_values(['dimension', 'date'])

        # Adjusting date to week starting Monday
        df_grp['date'] = pd.to_datetime(df_grp['date']) - pd.to_timedelta(6, unit='d')

        return df_grp

    def impute_missing_date(self, df_grp):

        """impute missing days with zero spend and target

        Returns:
            dataframe: dataframe with days imputed
        """
        if self.is_weekly_selected == True:
            days = 7
        else:
            days = 1

        df_day = pd.DataFrame()
        # _format = "%Y-%m-%d"
        for dim in df_grp["dimension"].unique():

            df_dim = df_grp[df_grp["dimension"] == dim]
        
            start = df_dim["date"].min()
            end = df_dim["date"].max()

            tmp = [start]

            while start < end:
                start += datetime.timedelta(days=days)
                tmp.append(start)

            df_tmp = pd.DataFrame()
            df_tmp["date"] = tmp
            df_tmp["dimension"] = dim
            df_day = df_day.append(df_tmp)

        df_grp = df_grp.merge(df_day, on=["dimension", "date"], how="outer")
        if self.use_impression == False:
            df_grp[["spend", "target"]] = df_grp[["spend", "target"]].fillna(0)
        else:
            df_grp[["spend", "impression", "target"]] = df_grp[["spend", "impression", "target"]].fillna(0)

        df_grp=df_grp.sort_values(['dimension', 'date']).reset_index(drop=True)

        return df_grp