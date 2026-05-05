import numpy as np
import pandas as pd
from scipy.stats import qmc


def generate_lhs_samples(reactants_config, n_samples=32, seed=42, output_file='samples.csv'):
    names = list(reactants_config.keys())
    d = len(names)
    log_lower_bound = [reactants_config[name][0] for name in names]
    log_upper_bound = [reactants_config[name][1] for name in names]

    sampler = qmc.LatinHypercube(d=d, seed=seed, optimization="random-cd")
    sample_unit = sampler.random(n=n_samples)
    sample_log = qmc.scale(sample_unit, log_lower_bound, log_upper_bound)

    if d == 1:
        sample_log = sample_log.reshape(-1, 1)

    sample_linear = 10 ** sample_log

    data_dict = {}
    for i, name in enumerate(names):
        data_dict[f'{name}_log10'] = sample_log[:, i]
        data_dict[name] = sample_linear[:, i]

    df = pd.DataFrame(data_dict)

    df.to_csv(output_file, index=False)
    print(f"成功生成 {n_samples} 个样本，维度为 {d}D，已保存至 {output_file}")
    return df


# config_1d = {
#     'add': [-3.0, 0.0]
# }
# df_1d = generate_lhs_samples(config_1d, n_samples=32, output_file='add.csv')
# print(df_1d.head())


print("-" * 30)


config_2d = {
    'TMB': [-2.301, 0.0],
    'H2O2': [-2.301, 0.0]
}

hrp_levels = [1.0, 0.01, 0.0001]
base_seed = 40

for i, hrp in enumerate(hrp_levels):
    file_name = f'./May_5_full_log/LHS_log10_HRP_{hrp}.csv'
    df = generate_lhs_samples(config_2d, n_samples=32, seed=base_seed + i, output_file=file_name)
    print(f"HRP={hrp} 组已生成，Seed={base_seed + i}")