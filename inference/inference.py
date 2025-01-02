import xgboost as xgb
import pandas as pd

class XGBoostInference:
    def __init__(self, model_path: str):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def prepare_data(self, input_data: str):
        input_df = pd.read_csv(input_data)
        columns = list(input_df.columns)
        columns.pop(columns.index("Class"))
        self.dTest = xgb.DMatrix(input_df[columns], input_df["Class"].values)

    def perform_inference(self):
        preds = self.model.predict(
            data=self.dTest,
            iteration_range=(0, self.model.best_iteration+1),
        )
        return preds