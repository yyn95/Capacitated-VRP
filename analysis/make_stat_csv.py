import pandas as pd
import json

def get_result_statistic_df(json_path):
    with open(json_path) as f:
        data = json.load(f)

    # remove algorithm and iteration
    data.pop("algorithm", None)
    data.pop("iteration", None)

    # transform the dictionary
    reform = dict()
    for out_key, inner_dict in data.items():
        for inner_key, v in inner_dict.items():
            reform[(out_key, inner_key)] = v

    df = pd.DataFrame.from_dict(reform)

    stat_df = df.describe(percentiles=[]).transpose()

    # count to integer
    stat_df['count'] = stat_df['count'].astype(int)

    # rename column names
    stat_df = stat_df.rename(lambda s: s.split(".")[0], axis="index")

    # sort index
    stat_df = stat_df.sort_index()

    return stat_df

if __name__ == '__main__':
    # GA algorithm
    ga_stat_df = get_result_statistic_df("benchmark_res/result_ga.json")
    ga_stat_df.to_csv("benchmark_res/result_stat_ga.csv")

    # SA algorithm
    sa_stat_df = get_result_statistic_df("benchmark_res/result_sa.json")
    sa_stat_df.to_csv("benchmark_res/result_stat_sa.csv")

    # tabu algorithm
    tabu_stat_df = get_result_statistic_df("benchmark_res/result_tabu.json")
    tabu_stat_df.to_csv("benchmark_res/result_stat_tabu.csv")
