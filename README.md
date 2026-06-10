# Env4Cast
This is the code for paper:
> Coupling Multi-Physics Processes and Multi-Scale Data Correlation Learning for Environmental Spatiotemporal Forecasting
> 
## Dependencies
Recent versions of the following packages for Python 3 are required:
* einops==0.7.0
* numpy==1.24.4
* pandas==2.0.3
* Pillow==10.1.0
* scikit-learn==1.3.2
* scipy==1.10.1
* torch==1.13.1
* torchaudio==0.10.1+cu111
* torchvision==0.11.2+cu111
* tornado==6.4
* tqdm==4.66.1

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* SST https://data.marine.copernicus.eu
* KnowAir https://github.com/shuowang-ai/PM2.5-GNN
* Beijing https://dataverse.harvard.edu/dataverse/whw195009



## Usage
Use the following command to run the main script:

python run.py

If you want to modify the experimental configuration parameters, please refer to run.py.
Specifically, key parameters such as the epochs, feature dimensions, and the number of dataset nodes can all be modified in the configuration file.

## Use your own dataset
Convert the data file into `.csv` format, place it in the `\dataset` directory, update the corresponding path in the `run.py` file, and modify the `num_nodes` in `run.py`.
