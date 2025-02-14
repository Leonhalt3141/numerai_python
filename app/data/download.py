import os

from numerapi import NumerAPI

save_path = "data"
if not os.path.exists(save_path):
    os.makedirs(save_path)


class Downloader:
    version = "v5.0"
    napi = NumerAPI()

    @classmethod
    def download_train(cls):
        train_path = f"{save_path}/train.parquet"
        if os.path.exists(train_path):
            os.remove(train_path)
        cls.napi.download_dataset(f"{cls.version}/train.parquet", train_path)

    @classmethod
    def download_validation(cls):
        validation_path = f"{save_path}/validation.parquet"
        if os.path.exists(validation_path):
            os.remove(validation_path)
        cls.napi.download_dataset(f"{cls.version}/validation.parquet", validation_path)

    @classmethod
    def download_live(cls):
        live_file = f"{save_path}/live.parquet"
        if os.path.exists(live_file):
            os.remove(live_file)
        cls.napi.download_dataset(f"{cls.version}/live.parquet", live_file)

    @classmethod
    def download_live_example(cls):
        live_example = f"{save_path}/live_example_preds.parquet"
        if os.path.exists(live_example):
            os.remove(live_example)
        cls.napi.download_dataset(
            f"{cls.version}/live_example_preds.parquet", live_example
        )

    @classmethod
    def download_validation_example(cls):
        validation_example = f"{save_path}/validation_example_preds.parquet"
        if os.path.exists(validation_example):
            os.remove(validation_example)
        cls.napi.download_dataset(
            f"{cls.version}/validation_example_preds.parquet", validation_example
        )

    @classmethod
    def download_features(cls):
        cls.napi.download_dataset(
            f"{cls.version}/features.json", f"{save_path}/features.json"
        )

    @classmethod
    def download_meta_model(cls):
        cls.napi.download_dataset(
            f"{cls.version}/meta_model.parquet", f"{save_path}/meta_model.parquet"
        )

    @classmethod
    def download_all(cls):
        print("Download train")
        cls.download_train()

        print("Download validation")
        cls.download_validation()

        print("Download live")
        cls.download_live()

        print("Download Live example")
        cls.download_live_example()

        print("Download Validation example")
        cls.download_validation_example()

        print("Download features")
        cls.download_features()

        print("Download meta model")
        cls.download_meta_model()
