## Documentation
### Parsing
Our approach learns from a symbolic representation (sheet music), which is parsed from **Guitar Pro** files.
To avoid compatibility issues between different versions, the **PyGuitarPro library** is used.
A custom parser expands the functionality of this library to generate a dataset of fixed-size sequences.
### Training
Machine Learning projects often require extensive hyperparameter tuning.
With this in mind, a generic script has been built to support several configurations of recurrent and feed-forward **Keras** models.
The Strategy pattern allows us to write a highly configurable training pipeline that reads the experiment conditions from a JSON file and trains custom models accordingly.


### Data viz
The metrics of each model are saved for every epoch as **Tensorboard** logs.
Tensorboard provides an intuitive dashboard comparing the metric graphs for a given set of models.
For the stages of the project that required interactive data manipulation, **Pandas** and **Seaborn** were used within a **Jupyter Notebook**.

For more details, read report.pdf
