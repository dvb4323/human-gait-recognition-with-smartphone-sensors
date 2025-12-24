# Human Gait Recognition

## For Developers:

### Prerequisites:

- Python 3.12.3
- Environment Management System (Anaconda)
- IDE: Pycharm
- For every new feature, create a branch and then create a pull request. Do not merge with the "master" branch without checks.

### First Start:

- Install the necessary packages: `pip install -r requirements.txt`
- Download the dataset, unzip it and place it in the `data/raw` directory

### Preprocessing:

- Run the `pipeline.py` script to preprocess the data.
- The preprocessed data will be saved in the `data/processed` directory.

### Training:

- Run the `train.py` script to train the 1D-CNN model.
- Run the `train_lstm.py` script to train the LSTM model.
- Run the `train_cnn_lstm.py` script to train the CNN-LSTM hybrid model.
- The trained models will be saved in the `results` directory, along with graphs and reports.

### More will be added in the future Development.
