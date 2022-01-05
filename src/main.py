import os
import json
import model_trainer


if __name__ == "__main__":
    with open(os.path.join("src", "config", "train_config.json"), "r") as fp:
        params = json.load(fp)

    mt = model_trainer.ModelTrainer()
    mt.run()
