import numerapi
import pandas as pd

# model_name = kuwaken
# public_id = "OFJRZ3M3QN33HZFM3AJBYFQEQTMS3ZUC"
# secret_key = "BU2HHSPCYZ3BEHL7F2QGCYBAMWPASIFALMMMKW7G456DI5NOTQAYBCABK5NKHJO4"

# model_name = "kuwaken_gbt"
public_id = "CJPYPIM5WY5237XKAWSDRVOP7E7LOSG3"
secret_key = "5AQIW55R77U7YXFBNGO6LZPUIFZ2HJOPCGOPB7MPFATJ63AUWDI457TA3YQRCZA4"


def submit(
    live,
    preds_live,
    valid=None,
    preds_valid=None,
    model_name="kuwaken_gbt",
    flag_submit="live",
):
    # prepare predictions_df
    df_sub_live = pd.DataFrame()
    df_sub_live["id"] = live.index
    df_sub_live["prediction"] = preds_live
    df_sub_live["prediction"] = df_sub_live["prediction"].rank(pct=True, method="first")
    df_sub_live.to_csv(f"preds_live_{model_name}.csv", index=False)

    # prepare predictions_df
    if valid is not None and preds_valid is not None:
        df_sub_valid = pd.DataFrame()
        df_sub_valid["id"] = valid.index
        df_sub_valid["prediction"] = preds_valid
        df_sub_valid["prediction"] = df_sub_valid["prediction"].rank(
            pct=True, method="first"
        )
        df_sub_valid.to_csv(f"preds_valid_{model_name}.csv", index=False)

    # Upload your predictions using API
    if flag_submit == "valid,live":
        napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
        model_id = napi.get_models()[model_name]
        submission_id = napi.upload_diagnostics(
            f"preds_valid_{model_name}.csv", model_id=model_id
        )
        print(submission_id)
        submission_id = napi.upload_predictions(
            f"preds_live_{model_name}.csv", model_id=model_id
        )
        print(submission_id)
    elif flag_submit == "valid":
        napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
        model_id = napi.get_models()[model_name]
        submission_id = napi.upload_diagnostics(
            f"preds_valid_{model_name}.csv", model_id=model_id
        )
        print(submission_id)
    elif flag_submit == "live":
        napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
        model_id = napi.get_models()[model_name]
        submission_id = napi.upload_predictions(
            f"preds_live_{model_name}.csv", model_id=model_id
        )
        print(submission_id)
