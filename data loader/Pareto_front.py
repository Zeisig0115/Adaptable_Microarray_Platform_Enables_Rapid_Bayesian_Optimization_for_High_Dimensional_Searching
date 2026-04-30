import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# ---------- Load ----------
df = pd.read_csv("data.csv")
# 这里假设 data.csv 里已经有三列：t_peak, I_max, D_blue
metrics = df[['t_peak', 'I_max', 'D_blue']].copy().dropna()
points = metrics.to_numpy()
N = len(points)

# ---------- Pareto (min t_peak, max I_max, max D_blue) ----------
is_pareto = np.ones(N, dtype=bool)

for i in range(N):
    if not is_pareto[i]:
        continue
    p = points[i]
    for j in range(N):
        if i == j:
            continue
        q = points[j]
        # j 支配 i 的条件：
        #   t_peak:  q <= p  (越小越好)
        #   I_max:   q >= p  (越大越好)
        #   D_blue:  q >= p  (越大越好)
        #   且至少一项严格更好
        if (
            (q[0] <= p[0]) and
            (q[1] >= p[1]) and
            (q[2] >= p[2]) and
            ((q[0] < p[0]) or (q[1] > p[1]) or (q[2] > p[2]))
        ):
            is_pareto[i] = False
            break

pareto_idx = metrics.index[is_pareto]
pf = metrics.loc[pareto_idx]

# ---------- Plot ----------
fig = plt.figure(figsize=(13, 11))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(metrics['t_peak'], metrics['I_max'], metrics['D_blue'],
           c='gray', alpha=0.3, s=25, label='All Wells')

ax.scatter(pf['t_peak'], pf['I_max'], pf['D_blue'],
           c='red', s=70, edgecolor='black', depthshade=True, label='Pareto Front')

ax.set_xlabel('t_peak (↓)', labelpad=10)
ax.set_ylabel('I_max (↑)', labelpad=10)
ax.set_zlabel('D_blue (↑)', labelpad=10)
ax.set_title('Pareto Front (min t_peak, max I_max, max D_blue)', pad=20)
ax.legend()
ax.view_init(elev=25, azim=-50)
plt.tight_layout()
plt.show()

# 可选：保存 Pareto 前沿数据（包含原始行信息）
# df.loc[pareto_idx].to_csv("pareto_front_data.csv", index=False)
