# Estimating K-means with Transformers

For the 2024-2025 Shang Lab ERSP project, we created a pipeline to train and test a transformer model on the results of the k-means algorithm to see if we could improve performance or overall accuracy.

## Installation

Use Python version 3.11 or 3.12 to ensure proper functionality of imported [TabPFN](https://pypi.org/project/tabpfn/) encoder. 

## Usage

Below is the sequence of actions to be taken when training and testing a model for k-means custering using our pipeline:

1. **Select relevant datasets off [OpenML](https://www.openml.org/).** OpenML is an open-source platform for sharing and organizing machine learning datasets, algorithms, and experiments. When selecting datasets, it is important to consider both the insights and challenges one may face with that dataset when used to train. 

    Some parameters of the dataset to take into consideration include (but are not limited to) number of features, number of data points, number of clusters, etc. It is important to note that the TabPFN encoder works best on datasets with *less than 5000 data points*, and this should be accounted for when searching for data.

2. **Preprocess data using `data_preprocessing_pipeline`.** This normalizes our data to improve training and testing outcomes.

3. **Obtain a cross-validation set using `cv_split.py`.** This allows our model to hypertune the parameters for training.

    One of the resources we used to evaluate the convergence of training data was [W&B](https://wandb.ai/site/), which provided specific graphs and metrics regarding the training rate of our model, and helped us tune our hyperparameters.

4. **Calculate optimal cluster centroids for given datasets.** We do this by passing our datasets through `Elki.ipynb`, which runs 20 variations of the k-means algorithm on the dataset, and selects the centroid with the lowest MSE loss.

    These variations of the k-means algorithm are from the [ELKI data mining framework](https://elki-project.github.io/), which is an open-source Java implementation of many well-known algorithms utilized for data mining.

5. **Pass processed data into training pipeline.** Further information on how we train our models and how you can run our pipeline can be found in `run_train_documentation.md`.

6. **Evaluate model using k-means variations.** Further information on how evaluation is done and how to utilize our testing methods can be found in `test_documentation.md`.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)