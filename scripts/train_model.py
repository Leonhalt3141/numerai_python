from app.model.NumeraiModel import NumeraiModel

if __name__ == "__main__":
    numerai_model = NumeraiModel()
    numerai_model.train_with_optuna()
