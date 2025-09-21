#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
import seaborn as sns


# In[2]:


def logexpectation(rho, px_theta):
    return -np.log(px_theta[0]*rho + px_theta[1]*(1-rho))

def expecationlog(rho, px_theta):
    return -rho*np.log(px_theta[0]) - (1-rho)*np.log(px_theta[1])

#E_rho(p(x|rho)^j)
def expecation(rho, px_theta,j):
    return rho*(px_theta[0])**j + (1-rho)*(px_theta[1])**j


def var(p,px_theta):
    mean = px_theta[0]*p + px_theta[1]*(1-p)
    mean_square = px_theta[0]*px_theta[0]*p + px_theta[1]*px_theta[1]*(1-p)
    return mean_square - mean*mean

def CE(rho,px,px_theta):
    return px[0]*logexpectation(rho,px_theta[0])+px[1]*logexpectation(rho,px_theta[1])

def ExpectedLogLoss(rho,px,px_theta):
    return px[0]*expecationlog(rho,px_theta[0])+px[1]*expecationlog(rho,px_theta[1])
    
def SecondOrderJensenBound(rho,px,px_theta):
    return ExpectedLogLoss(rho,px,px_theta) \
            - 0.5/(np.max(px_theta[0])*np.max(px_theta[0]))*px[0]*var(rho,px_theta[0]) \
            - 0.5/(np.max(px_theta[1])*np.max(px_theta[1]))*px[1]*var(rho,px_theta[1])

def b(k,j):
    return((-1)**j*comb(2*k-1,j)/j)

def lowerbound(rho,px_theta,k):
    lb=0
    for j in range(1,2*k):
        lb=lb+1/j+b(k,j)*expecation(rho,px_theta,j)*(expecation(rho,px_theta,1))**(-j)
    return lb    

print(lowerbound(0.4,(0.75,0.45),2))

def OurBound(rho,px,px_theta,k):
    return ExpectedLogLoss(rho=rho,px=px,px_theta=px_theta) \
            - px[0]*lowerbound(rho=rho,px_theta=px_theta[0],k=k) \
            - px[1]*lowerbound(rho=rho,px_theta=px_theta[1],k=k)


# In[3]:


px_theta = ((0.7, 0.45),(1-0.7, 1-0.45))   # pierwsza para to prawd. 1 w bi 1 i bi2, a druga prawd 0, rho to parametr mieszaniny
px = (0.6, 1-0.6) # prawd 1 i 0 w prawdziwym rozkladzie danych \ni


# In[4]:


rho= np.linspace(0.0,1.0,1001)
CE_rho = CE(rho,px,px_theta)
ELL_rho = ExpectedLogLoss(rho,px,px_theta)
LJ2_rho = SecondOrderJensenBound(rho,px,px_theta)
OurBound_rho_3order=OurBound(rho,px,px_theta,2)
OurBound_rho_5order=OurBound(rho,px,px_theta,3)
entropy = rho - px[0]*np.log(px[0]) - px[1]*np.log(px[1]) - rho


# In[5]:


px_theta_2 = ((0.6, 0.4),(1-0.6, 1-0.4))
px_2 = (0.6, 1-0.6)


# In[6]:


rho= np.linspace(0.0,1.0,1001)
CE_rho_2 = CE(rho,px_2,px_theta_2)
ELL_rho_2 = ExpectedLogLoss(rho,px_2,px_theta_2)
LJ2_rho_2 = SecondOrderJensenBound(rho,px_2,px_theta_2)
OurBound_rho_3order_2=OurBound(rho,px_2,px_theta_2,2)
OurBound_rho_5order_2=OurBound(rho,px_2,px_theta_2,3)
entropy_2 = rho - px_2[0]*np.log(px_2[0]) - px_2[1]*np.log(px_2[1]) - rho


# In[7]:


# Ustawienia stylu
sns.set_theme(style="whitegrid", palette="colorblind")
sns.set_context("paper", font_scale=2.2, rc={"lines.linewidth": 2, "lines.markersize": 7, "legend.markerscale": 1.5})

# Definicja znaczników osi Y
yticks_left = [0.67, 0.68, 0.69, 0.70, 0.71, 0.72]
yticks_right = [0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]

# Tworzenie wykresów
fig, ax = plt.subplots(2, 1, figsize=(14, 10)) #, sharex=True)

# Definiowanie stylów linii i markerów
line_styles = {
    'CE': ('-', 'o'),
    'ELL': (':', 's'),
    'LJ2': ('--', '^'),
    'Our3': ('--', 'D'),
    'Our5': ('--', 'p')
}
epsilon = 0.01

# ----- Lewy wykres: Model Misspecification -----
ax[0].plot(rho, CE_rho, label=r'Cross-entropy loss $CE(\varrho)$', 
           linestyle=line_styles['CE'][0], marker=line_styles['CE'][1], markevery=0.1)
ax[0].plot(rho, ELL_rho, label=r'Expected log-loss $\mathbb{E}_{\theta\!\sim\!\varrho}[L(\theta)]$', 
           linestyle=line_styles['ELL'][0], marker=line_styles['ELL'][1], markevery=0.1)
ax[0].plot(rho, LJ2_rho, label=r'Second order bound of [25]', # r'Second order bound of (Masegosa, 2020)', 
           linestyle=line_styles['LJ2'][0], marker=line_styles['LJ2'][1], markevery=0.1)
ax[0].plot(rho, OurBound_rho_3order, label=r'Our third order bound (i.e., k=2)', 
           linestyle=line_styles['Our3'][0], marker=line_styles['Our3'][1], markevery=0.1)
ax[0].plot(rho, OurBound_rho_5order, label=r'Our fifth order bound (i.e., k=3)', 
           linestyle=line_styles['Our5'][0], marker=line_styles['Our5'][1], markevery=0.1)

# Linie pionowe i tekst
minima = [
    (CE_rho, '', 1.001),
    (ELL_rho, '', 1.002),
    (LJ2_rho, '', 1.002),
    (OurBound_rho_3order, '', 1.004),
    (OurBound_rho_5order, '', 1.003)
]

for y_data, text, offset in minima:
    idx = np.argmin(y_data)
    x_pos = idx/(rho.shape[0]-1)
    ax[0].vlines(x_pos, 0.67, np.min(y_data), color='k', linestyle='--')
    ax[0].text(x_pos-0.01 if 'Our' not in text else x_pos-0.03, 
               np.min(y_data)*offset, text)

ax[0].set_xlim(0 - epsilon, 1 + epsilon)
ax[0].set_ylim(0.67, 0.72)
ax[0].set_yticks(yticks_left)
# ax[0].set_ylabel('Value')

# ----- Prawy wykres: Perfect Model Specification -----
ax[1].plot(rho, CE_rho_2, label=r'$CE(\varrho)$', 
           linestyle=line_styles['CE'][0], marker=line_styles['CE'][1], markevery=0.1)
ax[1].plot(rho, ELL_rho_2, label=r'$E_{\varrho}[L(\theta)]$', 
           linestyle=line_styles['ELL'][0], marker=line_styles['ELL'][1], markevery=0.1)
ax[1].plot(rho, LJ2_rho_2, label=r'Second order bound of (Masegosa, 2020)', 
           linestyle=line_styles['LJ2'][0], marker=line_styles['LJ2'][1], markevery=0.1)
ax[1].plot(rho, OurBound_rho_3order_2, label=r'Our third-order bound (i.e., k=2)', 
           linestyle=line_styles['Our3'][0], marker=line_styles['Our3'][1], markevery=0.1)
ax[1].plot(rho, OurBound_rho_5order_2, label=r'Our fifth-order bound (i.e., k=3)', 
           linestyle=line_styles['Our5'][0], marker=line_styles['Our5'][1], markevery=0.1)

# Linie pionowe i tekst
minima_2 = [
    (CE_rho_2, '', 1.002),
    (ELL_rho_2, '', 1.008),
    (LJ2_rho_2, '', 1.014),
    (OurBound_rho_3order_2, '', 1.020),
    (OurBound_rho_5order_2, '', 1.026)
]

for y_data, text, offset in minima_2:
    idx = np.argmin(y_data)
    x_pos = idx/(rho.shape[0]-1)
    ax[1].vlines(x_pos, 0.67, np.min(y_data), color='k', linestyle='--')
    ax[1].text(x_pos-0.01 if 'Our' not in text else x_pos-0.03, 
               np.min(y_data)*offset, text)

ax[1].set_xlim(0 - epsilon, 1 + epsilon)
ax[1].set_ylim(0.67, 0.76)
ax[1].set_yticks(yticks_right)
ax[1].set_xlabel(r'$\varrho$')
# ax[1].set_ylabel('Value')

# Wspólna legenda na górze
lines, labels = ax[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.52, 1.065))


plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Zostaw miejsce na legendę
# plt.savefig("Extended_Jensen.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()


# In[ ]:





# In[ ]:




