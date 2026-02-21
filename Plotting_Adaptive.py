import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

def load_and_aggregate(df):
    grouped = (
        df.groupby("Episode")
          .agg(
              Reward_mean     = ("FinalReward", "mean"),
              Reward_std      = ("FinalReward", "std"),
              CriticLoss_mean = ("CriticLoss", "mean"),
              CriticLoss_std  = ("CriticLoss", "std"),
              ActorLoss_mean  = ("ActorLoss", "mean"),
              ActorLoss_std   = ("ActorLoss", "std"),
          )
          .reset_index()
          .sort_values("Episode")
    )

    grouped = grouped.fillna(0.0)
    return grouped

# 1) CSV 읽기
# fc_df   = pd.read_csv("fc_m4_logs.csv")
# mh_df   = pd.read_csv("multihead_m4_logs.csv")
moe_df  = pd.read_csv("moe_m4_logs.csv")
adaptive_moe_df = pd.read_csv("a2cmoe_m4_adaptive_logs.csv")
# mmoe_df = pd.read_csv("mmoe_m4_logs.csv")

# 2) ★ 먼저 슬라이스를 적용한 뒤 ★
# 예시 1: 앞에서 150000행까지만 사용
N0 = 0
N1 = 150000
N2 = 300000
N3 = 450000
# fc_df_sub   = fc_df.iloc[N2:N3]
# mh_df_sub   = mh_df.iloc[N2:N3]
moe_df_sub  = moe_df.iloc[0:5000]
adaptive_moe_df_sub = adaptive_moe_df.iloc[0:5000]
# mmoe_df_sub = mmoe_df.iloc[N2:N3]

# EP_MAX = 15000
# fc_df_sub   = fc_df[fc_df["Episode"] <= EP_MAX]
# mh_df_sub   = mh_df[mh_df["Episode"] <= EP_MAX]
# moe_df_sub  = moe_df[moe_df["Episode"] <= EP_MAX]
# mmoe_df_sub = mmoe_df[mmoe_df["Episode"] <= EP_MAX]

# 3) DataFrame load_and_aggregate
# fc  = load_and_aggregate(fc_df_sub)
# mh   = load_and_aggregate(mh_df_sub)
moe  = load_and_aggregate(moe_df_sub)
adaptive_moe = load_and_aggregate(adaptive_moe_df_sub)
# mmoe = load_and_aggregate(mmoe_df_sub)
# ==========================================

def plot_ci(ax, x, mean, std, label, color):
    x    = np.asarray(x)
    mean = np.asarray(mean)
    std  = np.asarray(std)
    print(mean)
    ax.plot(x, mean, label=label, color=color)

fig, axs = plt.subplots(1, 3, figsize=(16, 6))
plt.suptitle("Comparison from CSV logs")

# 1) Reward
# plot_ci(axs[0], fc["Episode"],   fc["Reward_mean"],   fc["Reward_std"],
#         "FC", "tab:blue")
# plot_ci(axs[0], mh["Episode"],   mh["Reward_mean"],   mh["Reward_std"],
#         "MH", "tab:orange")
plot_ci(axs[0], moe["Episode"],  moe["Reward_mean"],  moe["Reward_std"],
        "MoE", "tab:green")
plot_ci(axs[0], adaptive_moe["Episode"], adaptive_moe["Reward_mean"], adaptive_moe["Reward_std"],
        "Adaptive MOE", "tab:red")
# plot_ci(axs[0], mmoe["Episode"], mmoe["Reward_mean"], mmoe["Reward_std"],
#         "MH + MOE", "tab:red")

axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")
axs[0].set_title("Reward Comparison")
axs[0].legend()
axs[0].grid(True)

# 2) Critic Loss
# plot_ci(axs[1], fc["Episode"],   fc["CriticLoss_mean"],   fc["CriticLoss_std"],
#         "FC", "tab:blue")
# plot_ci(axs[1], mh["Episode"],   mh["CriticLoss_mean"],   mh["CriticLoss_std"],
#         "MH", "tab:orange")
plot_ci(axs[1], moe["Episode"],  moe["CriticLoss_mean"],  moe["CriticLoss_std"],
        "MoE", "tab:green")
plot_ci(axs[1], adaptive_moe["Episode"], adaptive_moe["CriticLoss_mean"], adaptive_moe["CriticLoss_std"],
        "Adaptive MOE", "tab:red")
# plot_ci(axs[1], mmoe["Episode"], mmoe["CriticLoss_mean"], mmoe["CriticLoss_std"],
#         "MH + MOE", "tab:red")

axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Critic Loss")
axs[1].set_title("Critic Loss Comparison")
axs[1].legend()
axs[1].grid(True)

# 3) Actor Loss
#plot_ci(axs[2], fc["Episode"],   fc["ActorLoss_mean"],   fc["ActorLoss_std"], "FC", "tab:blue")
# plot_ci(axs[2], mh["Episode"],   mh["ActorLoss_mean"],   mh["ActorLoss_std"],
#         "MH", "tab:orange")
plot_ci(axs[2], moe["Episode"],  moe["ActorLoss_mean"],  moe["ActorLoss_std"],
        "MoE", "tab:green")
plot_ci(axs[2], adaptive_moe["Episode"], adaptive_moe["ActorLoss_mean"], adaptive_moe["ActorLoss_std"],
        "Adaptive MOE", "tab:red")
# plot_ci(axs[2], mmoe["Episode"], mmoe["ActorLoss_mean"], mmoe["ActorLoss_std"],
#         "MH + MOE", "tab:red")

axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Actor Loss")
axs[2].set_title("Actor Loss Comparison")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()