import pandas as pd
import numpy as np


class outlier_treatment:
    def __init__(self, df, use_impression):
        """
        intialization of the class
        """
        self.df = df
        self.use_impression = use_impression

    def z_outlier(self):
        """remove outlier using z score target and spend/impression

        Returns:
            dataframe: dataframe with outlier removed
        """

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        df_d = {}

        df_spend = self.df[self.df.spend != 0]

        for col in ["target", metric]:

            df_ = pd.DataFrame()

            for dim in self.df.dimension.unique():

                df__ = df_spend[df_spend["dimension"] == dim]

                df__.reset_index(drop=True, inplace=True)

                mean = np.mean(df__[col])

                std = np.std(df__[col])

                df__["z_outlier_" + col] = np.where(
                    abs((df__[col] - mean) / std) > 3, 1, 0
                )

                df__ = df__[df__["z_outlier_" + col] == 0]

                df__.drop(columns=["z_outlier_" + col], inplace=True)

                df_ = df_.append(df__, ignore_index=True)

            df_d[col] = df_

        df_sub = df_d[metric].merge(
            df_d["target"][["dimension", "date"]], on=["dimension", "date"], how="inner"
        )

        return df_sub

    def drop_points(self, df_sub):

        """count number of data dropped after outlier treatment

        Returns:
            dataframe: dataframe containing information about datapoint dropped
        """

        df_count_out = pd.DataFrame(df_sub["dimension"].value_counts()).rename(
            columns={"dimension": "data_points_post_outlier_treatment"}
        )

        df_count_ = pd.DataFrame(self.df["dimension"].value_counts()).rename(
            columns={"dimension": "no_of_data_points"}
        )

        df_drop = df_count_.merge(
            df_count_out, left_index=True, right_index=True, how="outer"
        ).fillna(0)

        df_drop["%_of_data_points_discarded_during_outlier_treatment"] = (
            (
                df_drop["no_of_data_points"]
                - df_drop["data_points_post_outlier_treatment"]
            )
            * 100
            / (df_drop["no_of_data_points"])
        ).round(decimals=2)

        return df_drop.reset_index().rename(columns={"index": "dimension"})
