SmallDatasetBenchmarks
======================
This repo is for testing machine learnign models on small (classification) datasets.

The relevant figures are produced in `figures.ipynb`. The actual experimental set-up is in files that start with `01_*`, `02_*`, etc. Nested cross-validation is used to get unbiased estimates of generalization performance. The splits are stratified random with fixed seeds, so the conclusions of these experiments are unlikely to hold for "real" data where test/production data is not IID with the training data. 

All that said, here are some observations:
- Non-linear models are better than linear ones, even for datas with < 100 samples. 
- SVM and Logistic Regression do similarly, but there are two datasets where SVM is the only algorithm that does not fail catastrophically. However, logistic regression with `elasticnet` penalty never gets less than 0.5 area under the ROC curve.
- LightGBM works well. Giving it more hyperparameters to try is a good idea. The `hyperopt` package did better than `scikit-optimize` and `Optuna` (not shown), but it could be user error.
- AutoGluon works really well and is the best approach for predictive power.  But you need to give it enough time. A 2m budget (per fold) was not enough, but 5m was enough for datasets up to 10k samples.

Data
----
The data is subset of this dataset-of-datasets: "UCI++, a huge collection of preprocessed datasets for supervised classification problems in ARFF format
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13748.svg)](http://dx.doi.org/10.5281/zenodo.13748)"

Note that UCI++ reuses the same datasets in different configurations and often you can't tell what's a categorical feature. 
