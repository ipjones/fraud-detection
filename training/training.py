import xgboost as xgb
import pandas as pd

MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50 #lgb early stop 
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result
RANDOM_STATE = 2000


class XGBoostTraining():
    def __init__(self):
        # initialize XGBoost parameters:
        self.params = {
            "objective":"binary:logistic",
            "eta": 0.039,
            "silent": True,
            "max_depth": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "eval_metric": 'auc',
            "random_state": RANDOM_STATE,
        }

    def initialize_dataset(self, train: str, val: str):
        train_df = pd.read_csv(train)
        val_df = pd.read_csv(val)

        # adjust data into xgb format
        columns = list(train_df.columns)
        columns.pop(columns.index("Class"))
        self.dTrain = xgb.DMatrix(data=train_df[columns], 
                                  label=train_df["Class"].values)
        self.dVal = xgb.DMatrix(data=val_df[columns],
                                label=val_df["Class"].values)

    def train_model(self):
        self.model = xgb.train(
            params=self.params,
            dtrain=self.dTrain,
            num_boost_round=MAX_ROUNDS,
            evals=[(self.dVal, 'validation')],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=VERBOSE_EVAL,
        )

    def export_model(self, out_path: str):
        self.model.save_model(out_path)