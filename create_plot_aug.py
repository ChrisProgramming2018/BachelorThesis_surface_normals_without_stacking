from create_plot import create_list, create_mean_and_std
import matplotlib.pyplot as plt
import glob
filenames = glob.glob("/home/programmer/12Semster/BA/experiments/results_surface_normals_augment_4_vs_8/0/*")
key_word = "Re"
min_idx = 4
max_idx = 10
key_word = "ep"
min_idx = 4
max_idx = 7
mean, var, length = create_mean_and_std(filenames, key_word, min_idx, max_idx, 4784)
filenames = glob.glob("/home/programmer/12Semster/BA/experiments/results_surface_normals_augment_4_vs_8/4/*")
mean1, var1, length1 = create_mean_and_std(filenames, key_word, min_idx, max_idx, min_length_p=length)
filenames = glob.glob("/home/programmer/12Semster/BA/experiments/results_surface_normals_augment_4_vs_8/8/*")
mean2, var2, length1 = create_mean_and_std(filenames, key_word, min_idx, max_idx, min_length_p=length)
filenames = glob.glob("/home/programmer/12Semster/BA/experiments/results_surface_normals_augment_4_vs_8/12/*")
mean3, var3, length1 = create_mean_and_std(filenames, key_word, min_idx, max_idx, min_length_p=length)
print(mean.shape, mean1.shape, mean2.shape, mean3.shape)


fig, axes = plt.subplots(1, 3, sharey=True)
x = [x for x in range(length)]

var_plus = mean + var 
var_minus = mean - var
var_minus[var_minus < 0] = 0

var_plus1 = mean1 + var1 
var_minus1 = mean1 - var1
var_minus1[var_minus1 < 0] = 0

var_plus2 = mean2 + var2 
var_minus2 = mean2 - var2
var_minus2[var_minus2 < 0] = 0

var_plus3 = mean3 + var3 
var_minus3 = mean3 - var3
var_minus3[var_minus3 < 0] = 0

fig.set_figwidth(20)
fig.set_figheight(20)
fig.suptitle("Comparing different crop sizes", fontsize=40)
# axes[0].title.set_text("crop size 0 vs 4")
axes[0].set_ylabel(r"mean goals of last 100 episodes", fontsize=20)
axes[0].set_xlabel("episodes", fontsize=20)
axes[0].plot(x, mean, color='r',label="random crop 0")
axes[0].fill_between(x, var_minus, var_plus, alpha=0.2, color='r')
axes[0].plot(x, mean1, color='b',label="random crop 4")
axes[0].fill_between(x, var_minus1, var_plus1, alpha=0.2, color='b')
axes[0].legend()
axes[0].legend(fontsize=20)



axes[1].set_ylabel(r"mean goals of last 100 episodes", fontsize=20)
axes[1].set_xlabel("episodes", fontsize=20)
axes[1].plot(x, mean1, color='r',label="random crop 4")
axes[1].fill_between(x, var_minus1, var_plus1, alpha=0.2, color='r')
axes[1].plot(x, mean2, color='b',label="random crop 8")
axes[1].fill_between(x, var_minus2, var_plus2, alpha=0.2, color='b')
axes[1].legend()
axes[1].legend(fontsize=20)

axes[2].set_ylabel(r"mean goals of last 100 episodes", fontsize=20)
axes[2].set_xlabel("episodes", fontsize=20)
axes[2].plot(x, mean2, color='r',label="random crop 8")
axes[2].fill_between(x, var_minus2, var_plus2, alpha=0.2, color='r')
axes[2].plot(x, mean3, color='b',label="random crop 12")
axes[2].fill_between(x, var_minus3, var_plus3, alpha=0.2, color='b')
axes[2].legend()
axes[2].legend(fontsize=20)




"""
axes[0,0].set_ylabel(r"mean goal of last 100 episodes")
axes[0,0].set_xlabel("episodes")
axes[0,0].plot(x, mean, color='r',label="random crop 0")
axes[0,0].fill_between(x, var_minus, var_plus, alpha=0.2, color='r')
axes[0,0].plot(x, mean1, color='b',label="random crop 4")
axes[0,0].fill_between(x, var_minus1, var_plus1, alpha=0.2, color='b')
axes[0,0].legend()
axes[0,0].legend(fontsize=20)

axes[0,1].set_ylabel(r"mean goal of last 100 episodes")
axes[0,1].set_xlabel("episodes")
axes[0,1].plot(x, mean1, color='r',label="random crop 4")
axes[0,1].fill_between(x, var_minus1, var_plus1, alpha=0.2, color='r')
axes[0,1].plot(x, mean2, color='b',label="random crop 8")
axes[0,1].fill_between(x, var_minus2, var_plus2, alpha=0.2, color='b')
axes[0,1].legend()
axes[0,1].legend(fontsize=20)


axes[1,0].set_ylabel(r"mean goal of last 100 episodes")
axes[1,0].set_xlabel("episodes")
axes[1,0].plot(x, mean2, color='r',label="random crop 8")
axes[1,0].fill_between(x, var_minus2, var_plus2, alpha=0.2, color='r')
axes[1,0].plot(x, mean3, color='b',label="random crop 12")
axes[1,0].fill_between(x, var_minus3, var_plus3, alpha=0.2, color='b')
axes[1,0].legend()
axes[1,0].legend(fontsize=20)

axes[1,1].set_ylabel(r"mean goal of last 100 episodes")
axes[1,1].set_xlabel("episodes")
axes[1,1].plot(x, mean, color='r',label="random crop 0")
axes[1,1].fill_between(x, var_minus, var_plus, alpha=0.2, color='r')
axes[1,1].plot(x, mean1, color='b',label="random crop 4")
axes[1,1].fill_between(x, var_minus1, var_plus1, alpha=0.2, color='b')
axes[1,1].legend()
axes[1,1].legend(fontsize=20)
"""






plt.tight_layout()
plt.show()
