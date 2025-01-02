import random
import numpy

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
INPUT_FILE = "creditcard.csv"

class DataSplitter:
    def __init__(self, train: float, val: float, test: float, random_seed: int):
        if random_seed == None:
            random_seed = 0
        random.seed(random_seed)

        if not numpy.isclose(train+val+test, 1):
            print ("Error, train/test/val split does not sum to 1.")
        else:
            self.train = train
            self.val = val
            self.test = test

    def split_csv(self, infile: str):
        ipf = open(infile, mode="r", encoding="utf-8")

        line = ipf.readline().strip()
        train = open(f"train_{infile}", "w")
        test = open(f"test_{infile}", "w")
        val = open (f"val_{infile}", "w")
        # Write header line to all files.
        train.write(line + "\n")
        val.write(line + "\n")
        test.write(line + "\n")

        line = ipf.readline().strip()

        while (line):
            roll = random.random()
            if roll < self.train:
                train.write(line + "\n")
            elif roll < self.train + self.val:
                val.write(line + "\n")
            else:
                test.write(line + "\n")
            
            line = ipf.readline().strip()

        train.close()
        test.close()
        val.close()


data_split = DataSplitter(train=TRAIN_SPLIT,
                          val=VAL_SPLIT,
                          test=TEST_SPLIT,
                          random_seed=0)
data_split.split_csv(infile=INPUT_FILE)