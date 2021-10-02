from src.ConfigManager import ConfigManager
from src.ModelBenchmark import ModelBenchmark
from src.DatasetFromCSV import DatasetFromCSV

if __name__ == "__main__":
    config = ConfigManager()
    dataset = DatasetFromCSV(config.input_notes, config.output_notes, config.input_dir)

    benchmark = ModelBenchmark(config, dataset)
    benchmark.execute()
