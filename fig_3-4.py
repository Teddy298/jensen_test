#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# In[2]:


df_lognormal = pd.read_csv('lognormal_20000.csv')
display(df_lognormal)
df_gamma = pd.read_csv('gamma_20000.csv')
display(df_lognormal)


# In[3]:


import matplotlib.ticker as ticker


sns.set_theme(style="whitegrid", palette="colorblind")
sns.set_context("paper", font_scale=3.6, rc={'lines.linewidth': 2, 'legend.fontsize': 35})

colors = sns.color_palette("colorblind")

x_lognormal = "sigma"
x_gamma = "a"
old_label = "Struski et al. (2023)"
old_label = "Bounds of [34]"
lab_k2 = "Our bounds for k=2"
lab_k3 = "Our bounds for k=3"

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24, 8), sharex='col') #, sharex=True, sharey=True)
df_lognormal.plot(x=x_lognormal, y="sigma.2", color='k', label=old_label, ax=axs[0, 0], legend=False)
df_lognormal.plot(x=x_lognormal, y="ub_k2", color=colors[1], label=lab_k2, ax=axs[0, 0], legend=False)
df_lognormal.plot(x=x_lognormal, y="ub_k3", color=colors[0], label=lab_k3, ax=axs[0, 0], legend=False)

axs[0, 0].legend(loc="lower center", bbox_to_anchor=(0.55, 1.02, 1, 0.2), #mode="expand", 
              borderaxespad=0, ncol=5, fancybox=True, shadow=True)
axs[0, 0].set_ylabel("Upper bound")
axs[0, 0].set_xlim(left=.1, right=0.38)
axs[0, 0].set_ylim(bottom=-0.01, top=0.21)

axs[1, 0].plot(df_lognormal[x_lognormal], np.zeros_like(df_lognormal[x_lognormal]), color='k',  label="old")
df_lognormal.plot(x=x_lognormal, y="lb_k2", color=colors[1], label="k=2", ax=axs[1, 0], legend=False)
df_lognormal.plot(x=x_lognormal, y="lb_k3", color=colors[0], label="k=3", ax=axs[1, 0], legend=False)
axs[1, 0].set_ylabel("Lower bound")
axs[1, 0].set_xlabel(fr'Value of parameter $\sigma$')

axs[1, 0].set_xlim(left=.1, right=0.38)
axs[1, 0].set_ylim(bottom=-0.004, top=.06)

#  =========================================================

df_gamma.plot(x=x_gamma, y="log(a/a-1)", color='k', label=old_label, ax=axs[0, 1], legend=False)
df_gamma.plot(x=x_gamma, y="ub_k2", color=colors[1], label="k=2", ax=axs[0, 1], legend=False)
df_gamma.plot(x=x_gamma, y="ub_k3", color=colors[0], label="k=3", ax=axs[0, 1], legend=False)
# axs[0, 1].set_ylabel("Upper bound")
axs[0, 1].set_xlim(left=9.5, right=16)

axs[1, 1].plot(df_gamma[x_gamma], np.zeros_like(df_gamma[x_gamma]), color='k',  label="old")
df_gamma.plot(x=x_gamma, y="lb_k2", color=colors[1], label="k=2", ax=axs[1, 1], legend=False)
df_gamma.plot(x=x_gamma, y="lb_k3", color=colors[0], label="k=3", ax=axs[1, 1], legend=False)

axs[1, 1].set_xlim(left=9.5, right=16)
# axs[1, 1].set_ylabel("Lower bound")
axs[1, 1].set_xlabel(f'Value of parameter $a$')

axs[1, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# plt.figtext(0.49, 0.51, "Lower bounds", rotation=90, ha="center", va="center") #, fontsize=16)

plt.subplots_adjust(
    left=0.1,
    bottom=0.1, 
    right=0.9, 
    top=0.9, 
#     wspace=0.12, 
    wspace=0.13, 
    hspace=0.1
)

# plt.savefig("theoretical_synthetic.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()


# # Barplot

# In[4]:


data = {
    "dataset": ["MNIST"] * 3 + ["SVHN"] * 3 + ["MNIST"] * 3 + ["SVHN"] * 3,
    "model": ["VAE", "IWAE-5", "IWAE-10"] * 4,
    "val": [
        0.09453452388192028, 0.005629009509085525, 0.005318140078021879,
        3.0076198541335386, 1.8248722557467678, 2.1639316453067563,
        0.2345964392313212, 0.07489271588287376, 0.07152908382886852,
        3.0120064826625765, 1.9223494105581365, 2.247819241146866
    ],
    "lab": ["Optimal bounds"] * 3 + ["Our bounds"] * 3 + ["Bounds of Struski et al. (2023)"] * 6,
#     "color": [colors[0]] * 3 + [colors[1]] * 3 + [colors[2]] * 6,
    "val_best": [
        0.03704849231339273, 0.005629009509085525, 0.005318140078021879,
        3.398588150234546, 0.18007955212456442, 0.22658720200943797,
        0.1823373612430173, 0.07489271588287376, 0.07152908382886852,
        4.1346495474799205, 0.3913226392550997, 0.4535636589718361
    ]
}


df = pd.DataFrame.from_dict(data)
display(df)


df.loc[len(df)] = {'dataset': 'CelebA', 'model': "VAE", 'val': np.nan, 
                   "lab": "Our bounds", "val_best": 0.12153520206464402}
df.loc[len(df)] = {'dataset': 'CelebA', 'model': "VAE", 'val': np.nan, 
                   "lab": "Bounds of [34]", #"Bounds of Struski et al. (2023)", 
                   "val_best": 0.1945610297843814}

df.loc[len(df)] = {'dataset': 'CelebA', 'model': "IWAE-5", 'val': np.nan, 
                   "lab": "Our bounds", "val_best": 0.008951464968435395}
df.loc[len(df)] = {'dataset': 'CelebA', 'model': "IWAE-5", 'val': np.nan, 
                   "lab": "Bounds of [34]", #"Bounds of Struski et al. (2023)", 
                   "val_best": 0.053160996759554956}

df.loc[len(df)] = {'dataset': 'CelebA', 'model': "IWAE-10", 'val': np.nan, 
                   "lab": "Our bounds", "val_best": 0.008305903578344221}
df.loc[len(df)] = {'dataset': 'CelebA', 'model': "IWAE-10", 'val': np.nan, 
                   "lab": "Bounds of [34]", #"Bounds of Struski et al. (2023)", 
                   "val_best": 0.05255227628154586}

df


# In[5]:


sns.set_theme(style="whitegrid", palette="colorblind")
sns.set_context("paper", font_scale=2.5, rc={'lines.linewidth': 2, 'legend.fontsize': 24})

colors = sns.color_palette("colorblind")


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 6)) #, sharey='row', sharex='col') #, sharex=True, sharey=True)

df_mnist = df.loc[(df['dataset'] == "MNIST")]
# display(df_mnist)
sns.barplot(df_mnist, x="model", y="val_best", hue="lab", 
            errorbar=None, color='blue', 
            palette=['tab:green', 'tab:orange'], ax=axs[0])

# axs[0].legend(loc="lower center", bbox_to_anchor=(0.55, 1.02, 1, 0.2), #mode="expand", 
#               borderaxespad=0, ncol=5, fancybox=True, shadow=True)
axs[0].set_xlabel("")
axs[0].set_ylabel("Upper bound")
axs[0].legend_.remove()

df_svhn = df.loc[(df['dataset'] == "SVHN")]
# display(df_svhn)
sns.barplot(df_svhn, x="model", y="val_best", hue="lab", 
            errorbar=None, color='blue', 
            palette=['tab:green', 'tab:orange'], ax=axs[1])

axs[1].set_xlabel("")
axs[1].set_ylabel("")
axs[1].legend_.remove()

df_celeba = df.loc[(df['dataset'] == "CelebA")]
# display(df_svhn)
sns.barplot(df_celeba, x="model", y="val_best", hue="lab", 
            errorbar=None, color='blue', 
            palette=['tab:green', 'tab:orange'], ax=axs[2])

axs[2].set_xlabel("")
axs[2].set_ylabel("")
axs[2].legend_.remove()


plt.subplots_adjust(
    left=0.1,
    bottom=0.1, 
    right=0.9, 
    top=0.9, 
    wspace=0.24, 
    hspace=0.1
)

lines = [] 
labels = [] 
  
for ax in fig.axes: 
    Line, Label = ax.get_legend_handles_labels() 
    # print(Label) 
    lines.extend(Line) 
    labels.extend(Label) 

print(labels)
    
# lines = list(set(lines)) 
# labels = list(set(labels))
    
lines = [lines[i] for i  in [-2, -1]]
labels = [labels[i] for i  in [-2, -1]]

axs[0].tick_params(axis='x', rotation=20)
axs[1].tick_params(axis='x', rotation=20)
axs[2].tick_params(axis='x', rotation=20)
    
# fig.legend(lines, labels, loc='upper right')
fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0., .93, 1, 0.2), #mode="expand", 
              borderaxespad=0, ncol=2, fancybox=True, shadow=True)

# plt.savefig("barplot_results.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()


# In[ ]:





# In[ ]:




