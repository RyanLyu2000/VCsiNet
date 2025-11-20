import numpy as np
loaded_data = np.load("/media/D/tankailin/NTSCC_analysis/NTC_CIFAR10_linear/history/NTC/[NTC_lmbda=1.6] 2024-01-07 02:11:27_test/data4analysis.npy", allow_pickle=True).item()
print(loaded_data.keys())

print(loaded_data["y_mean"].shape)      #means of y (from statistics)
print(loaded_data["y_cov_matrix"].shape)  #covariance matrices of y (from statistics)
print(loaded_data["w_mean"].shape)     #means of w (from statistics)
print(loaded_data["w_std"].shape)      #means of w (from statistics)
print(loaded_data["w_cov_matrix"].shape)    #covariance matrices of w (from statistics)
print(loaded_data["w_mean_learned"].shape)   #learned means of w
print(loaded_data["w_std_learned"].shape)    #learned stds of w
print(loaded_data["H_matrix_learned"].shape)   #learned linear layer

mask = np.tril(np.ones((64, 64)))-np.eye(64)
print(mask)
tril_H = loaded_data["H_matrix_learned"] * mask
print(tril_H[0])
