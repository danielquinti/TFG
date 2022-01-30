import os
import sys
sys.path.append(
    os.path.join(
        "src",
        "architecture"
    )
)
sys.path.append(
    os.path.join(
        "src",
        "data_processing"
    )
)
import model_trainer
from data_processing import song_splitter

if __name__ == "__main__":
    # song_splitter.split_songs()
    mt = model_trainer.ModelTrainer()
    mt.run_all()
