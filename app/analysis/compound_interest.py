##
import datetime

import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt


def calc_monthly_compound_interest(
    principal: float = 0.0,
    monthly_deposit: float = 300000.0,
    interest_rate: float = 0.04,
    year: int = 5,
    start_year: int = 2024,
) -> pd.DataFrame:
    monthly_interest_rate = interest_rate / 12.0

    dates = []

    for y in range(year):
        dates += [datetime.datetime(start_year + y, m, 1) for m in range(1, 13)]

    df = pd.DataFrame({"date": dates})

    principal_list = [
        monthly_deposit * i + principal for i, date in enumerate(dates, 1)
    ]

    interest_list = []
    for i, p in enumerate(principal_list):
        if i == 0:
            interest_list.append(p * monthly_interest_rate)
        else:
            previous_interest = interest_list[i - 1]
            interest_list.append(
                (p + previous_interest) * monthly_interest_rate + previous_interest
            )

    real_interest_rate_list = [
        i / p if p != 0 else 0 for p, i in zip(principal_list, interest_list)
    ]

    df["principal"] = principal_list
    df["interest"] = interest_list
    df["real_interest_rate"] = real_interest_rate_list
    df["total_asset"] = df["principal"] + df["interest"]

    return df


def calc_monthly_cut_down(principal: float, cut_down, interest_rate, year, start_year):
    # dates = []
    monthly_interest_rate = interest_rate / 12

    # for y in range(year):
    #     dates += [datetime.datetime(start_year + y, m, 1) for m in range(1, 13)]
    #
    # df = pd.DataFrame({"date": dates})
    # principal_list = [principal - cut_down * i for i, date in enumerate(dates, 1)]
    #
    #
    # interest_list = []
    # for i, p in enumerate(principal_list):
    #     if i == 0:
    #         interest_list.append(p * monthly_interest_rate)
    #     else:
    #         previous_interest = interest_list[i - 1]
    #         interest_list.append(
    #             (p + previous_interest) * monthly_interest_rate + previous_interest
    #         )
    #
    # df["principal"] = principal_list
    # df["interest"] = interest_list
    # df["total_asset"] = df["principal"] + df["interest"]

    dates = []
    interest_list = []
    principal_list = []
    total_asset_list = []
    i = 0
    while (principal > 0) and (year >= i // 12):
        if i == 0:
            principal -= cut_down

        else:
            principal = principal - cut_down + interest_list[i - 1]

        interest = principal * monthly_interest_rate
        total_asset = principal + interest

        principal_list.append(principal)
        interest_list.append(interest)
        total_asset_list.append(total_asset)
        i += 1

    for y in range(len(total_asset_list) // 12 + 1):
        dates += [datetime.datetime(start_year + y, m, 1) for m in range(1, 13)]

    dates = dates[: len(total_asset_list)]
    df = pd.DataFrame(
        {
            "date": dates,
            "principal": principal_list,
            "interest": interest_list,
            "total_asset": total_asset_list,
        }
    )

    return df


# def nisa_simulation():
ci_df = calc_monthly_compound_interest(0)
nisa_principal = ci_df.tail(1)["principal"].values[0]
nisa_total_asset = ci_df.tail(1)["total_asset"].values[0]

nisa_df = calc_monthly_compound_interest(
    nisa_total_asset,
    0,
    0.04,
    30,
    ci_df.tail(1)["date"].dt.to_pydatetime()[0].year + 1,
)
nisa_df["principal"] = nisa_principal
nisa_df["interest"] = nisa_df["total_asset"] - nisa_df["principal"]
ci_df = pd.concat([ci_df, nisa_df])
##
ci_df.set_index("date", inplace=True)
ci_df["total_asset"] = ci_df["total_asset"] / 10000.0
ci_df["principal"] = ci_df["principal"] / 10000.0
ci_df["interest"] = ci_df["interest"] / 10000

sn.set_theme()
fig = plt.figure(figsize=(10, 5))
plt.stackplot(
    ci_df.index, ci_df["principal"], ci_df["interest"], labels=["Principal", "Interest"]
)
plt.plot(ci_df.index, ci_df["total_asset"], lw=3, c="k")
plt.ylabel("Ten Thousand JPY", fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()

pv = ci_df.tail(1)["total_asset"].values[0] * 10000
cd_df = calc_monthly_cut_down(pv, 150000, 0.04, 20, 2034)
cd_df["total_asset"] /= 10000

sn.set_theme()
fig = plt.figure(figsize=(10, 5))
plt.plot(cd_df["date"], cd_df["total_asset"])
plt.ylabel("Ten Thousand JPY", fontsize=10)
plt.tight_layout()
plt.show()


# nisa_simulation()
##
