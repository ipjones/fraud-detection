from training.training import XGBoostTraining
from inference.inference import XGBoostInference
# This is a very simple coordinator script.

# MODE = "TRAIN"
MODE = "TRAIN_AND_INFER"
# MODE = "INFER"
TRAIN_DATA = "data/train_creditcard.csv"
VAL_DATA = "data/val_creditcard.csv"
TEST_DATA = "data/test_creditcard.csv"
MODEL_PATH = "models/xgboost_model.xgb"

def train_and_infer():
    if MODE == "TRAIN" or MODE == "TRAIN_AND_INFER":
        trainer = XGBoostTraining()
        trainer.initialize_dataset(
            train=TRAIN_DATA,
            val=VAL_DATA
        )
        trainer.train_model()
        trainer.export_model(MODEL_PATH)
    if MODE == "TRAIN_AND_INFER" or MODE == "INFER":
        infer = XGBoostInference(MODEL_PATH)
        infer.prepare_data(TEST_DATA)
        print (infer.perform_inference())
    return

if __name__ == "__main__":
    train_and_infer()