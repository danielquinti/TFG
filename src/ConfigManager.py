import json
class ConfigManager():
    def __init__(self):
        with open("src\\config.json") as file:
            params = json.load(file)
            self.input_notes = params["INPUT_NOTES"]
            self.output_notes = params["OUTPUT_NOTES"]
            self.output_dir = params["OUTPUT_DIR"]
            self.batch_size = params["BATCH_SIZE"]
            self.max_epochs = params["MAX_EPOCHS"]
            self.input_dir = params["INPUT_DIRS"][params["DUMMY"]]
            self.selected_models = params["SELECTED_MODELS"]
            self.validation_split = params["VALIDATION_SPLIT"]
            self.loss_weights = params["LOSS_WEIGHTS"]