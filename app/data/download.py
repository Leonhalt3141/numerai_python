from numerapi import NumerAPI


class Downloader:
    napi = NumerAPI()

    @classmethod
    def download_train(cls):
        cls.napi.download_dataset("v4.1/train_int8.parquet", "train.parquet")

    @classmethod
    def download_validation(cls):
        cls.napi.download_dataset("v4.1/validation_int8.parquet", "validation.parquet")

    @classmethod
    def download_live(cls):
        cls.napi.download_dataset("v4.1/live_int8.parquet", "live.parquet")

    @classmethod
    def download_live_example(cls):
        cls.napi.download_dataset(
            "v4.1/live_example_preds.parquet", "live_example_preds.parquet"
        )

    @classmethod
    def download_validation_example(cls):
        cls.napi.download_dataset(
            "v4.1/validation_example_preds.parquet", "validation_example_preds.parquet"
        )

    @classmethod
    def download_features(cls):
        cls.napi.download_dataset("v4.1/features.json", "features.json")

    @classmethod
    def download_meta_model(cls):
        cls.napi.download_dataset("v4.1/meta_model.parquet", "meta_model.parquet")

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
