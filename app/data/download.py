import os

from numerapi import NumerAPI


class Downloader:
    version = "v4.2"
    napi = NumerAPI()

    @classmethod
    def download_train(cls):
        train_path = "train.parquet"
        if os.path.exists(train_path):
            os.remove(train_path)
        cls.napi.download_dataset(f"{cls.version}/train_int8.parquet", train_path)

    @classmethod
    def download_validation(cls):
        validation_path = "validation.parquet"
        if os.path.exists(validation_path):
            os.remove(validation_path)
        cls.napi.download_dataset(
            f"{cls.version}/validation_int8.parquet", validation_path
        )

    @classmethod
    def download_live(cls):
        live_file = "live.parquet"
        if os.path.exists(live_file):
            os.remove(live_file)
        cls.napi.download_dataset(f"{cls.version}/live_int8.parquet", live_file)

    @classmethod
    def download_live_example(cls):
        live_example = "live_example_preds.parquet"
        if os.path.exists(live_example):
            os.remove(live_example)
        cls.napi.download_dataset(
            f"{cls.version}/live_example_preds.parquet", live_example
        )

    @classmethod
    def download_validation_example(cls):
        validation_example = "validation_example_preds.parquet"
        if os.path.exists(validation_example):
            os.remove(validation_example)
        cls.napi.download_dataset(
            f"{cls.version}/validation_example_preds.parquet", validation_example
        )

    @classmethod
    def download_features(cls):
        cls.napi.download_dataset(f"{cls.version}/features.json", "features.json")

    @classmethod
    def download_meta_model(cls):
        cls.napi.download_dataset(
            f"{cls.version}/meta_model.parquet", "meta_model.parquet"
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
