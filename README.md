# numerai_python
This repository contains my personal machine learning models for the Numerai competition.

### Features
* Retrieve training data using the Numerai API
* Train machine learning models using the training data
* Generate predictions on live and validation data using trained models
* Upload prediction results to Numerai and reflect the outcomes

### Model Versioning

Model versions are managed by numbering the script files.
For example, app/train/train002.py trains version 2 of the model.
To run training in the Python console:

```python
from app.train.train002 import main
main()
```

To submit the prediction results:
```python
from app.submit.submit002 import submission

submission()
```

### Model Descriptions
Model 002 uses a training pipeline built with scikit-learn and Optuna.
The pipeline includes various types of data transformers, estimators, and neutralization procedures, implemented in modular stages.
Optuna, a Bayesian optimization library, is used to find the optimal combination of components.
Available estimators include Random Forest and XGBoost.

Model 003 also uses scikit-learn and Optuna, but introduces an ensemble approach.
Numerai provides sub-targets as auxiliary prediction targets.
This model trains a LightGBM model on sub-targets and ensembles the outputs to derive the final prediction.

### Future Work

I plan to continue exploring and implementing various approaches to improve prediction performance.