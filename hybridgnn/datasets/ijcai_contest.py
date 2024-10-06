
import pandas as pd
import os
import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)

from relbench.base import Database, Dataset, Table


class IJCAIContestDataset(Dataset):
    """Original data source:
    https://tianchi.aliyun.com/dataset/42"""

    val_timestamp = pd.Timestamp("2014-09-30")
    test_timestamp = pd.Timestamp("2014-10-15")
    url = "https://tianchi.aliyun.com/dataset/42"
    err_msg = (
        "{data} not found. Please download "
        "data_format1.zip from '{url}', "
        "move it to '{path}' and unzip."
    )

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""

        def time_to_date(t):
            t_str = str(t)
            if len(t_str) == 3:
                t_str = "0" + t_str
            mm, dd = t_str[:2], t_str[2:]
            return f"2014-{mm}-{dd}"

        path = os.path.join("data", "IJCAI-15-Repeat-Buyers-Prediction-Dataset")
        users = pd.read_csv(os.path.join(path, "user_info_format1.csv"))
        logs = pd.read_csv(os.path.join(path, "user_log_format1.csv"))
        #train = pd.read_csv(os.path.join(path, "train_format1.csv"))
        #test = pd.read_csv(os.path.join(path, "test_format1.csv"))
        logs['time_stamp'] = pd.to_datetime(logs['time_stamp'].apply(time_to_date), format='%Y-%m-%d')

        logs.rename(columns={'seller_id': 'merchant_id'}, inplace=True)

        db = Database(
            table_dict={
                "users": Table(
                    df=users,
                    fkey_col_to_pkey_table={},
                    pkey_col="user_id",
                    time_col=None,
                ),
                "logs": Table(
                    df=logs,
                    fkey_col_to_pkey_table={
                        "user_id": "users",
                    },
                    pkey_col=None,
                    time_col="time_stamp",
                )
            }
        )

        db = db.upto(pd.Timestamp("2014-10-30"))
        return db

class UserItemPurchaseTask(RecommendationTask):
    r"""Predict the list of distinct items each customer will purchase."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "item_id"
    dst_entity_table = "logs"
    time_col = "time_stamp"
    timedelta = pd.Timedelta(days=15)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        logs = db.table_dict["logs"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    users.user_id,
                    LIST(DISTINCT logs.item_id) AS item_id
                FROM
                    timestamp_df t
                LEFT JOIN
                    logs
                ON
                    logs.item_id > t.timestamp AND
                    logs.time_stamp <= t.timestamp + INTERVAL '{self.timedelta} days'
                GROUP BY
                    t.timestamp,
                    logs.user_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )