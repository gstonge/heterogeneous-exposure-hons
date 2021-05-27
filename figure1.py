import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

color_list = ["#7fcdbb","#1d91c0","#0c2c84","#d4748b","#c96942","#deb440"]
lw = 1.

#plot parameters
font_size=8
plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font',family='serif',serif='Computer Modern')
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('legend', fontsize=font_size)

#plot
width = 7.057/2.
height = width/1.4
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(width,height))

plt.subplots_adjust(left=0.25, bottom=0.155, right=0.8,
                    top=0.80, wspace=0.38, hspace=0.25)

df = pd.read_csv('SSEs.dat', delim_whitespace=True, header=None)

type_merged = ['transport','work']

xvec = np.array(df[0])
yvec = np.array(df[3])
tvec = np.array(df[5])
svec = 200*np.array(df[2])
type_list = sorted(list(set(df[5])))
for t in type_merged:
    type_list.remove(t)
type_list.append('other')
color_map = {t:c for t,c in zip(type_list,color_list)}
print(color_map)

for x,y,s,t in zip(xvec,yvec,svec,tvec):
    if not t in type_merged:
        ax.scatter(x,y,s=s,color=color_map[t], facecolors='none',
                  linewidths=lw)
    else:
        ax.scatter(x,y,s=s,color=color_map['other'], facecolors='none',
                  linewidths=lw)


box = ax.get_position()
ax.set_position([box.x0, box.y0,
                 box.width, box.height * 0.9])

#legend 1
l_list = [plt.scatter([],[], s=s,color='#1a1a1a', facecolors='none',
                  linewidths=lw) for s in [2,20,200]]
labels = ["1 \%", "10 \%", "100 \% Attack rate"]

leg1 = plt.legend(l_list, labels, ncol=3, fontsize=font_size,
                 loc='upper center', bbox_to_anchor=(0.5, 1.22),
                 frameon=False,handletextpad=0.1,
                 columnspacing=0.7)
#legend 2
for t in type_list:
    if not t in type_merged:
        l = plt.scatter([],[], s=20, marker='s', color=color_map[t], label=t)
    else:
        l = plt.scatter([],[], s=20, marker='s', color=color_map['other'], label='other')
leg2 = plt.legend(ncol=3, fontsize=font_size,
                 loc='upper center', bbox_to_anchor=(0.5, 1.5),
                 frameon=False,handletextpad=0.1)

# Axes label.
ax.set_xlabel(r'Duration [hours]')
ax.set_ylabel(r'Size [people]')

ax.set_xscale('log')
ax.set_yscale('log')

# Bounds.
ax.set_xbound(1.0,1500)
ax.set_ybound(3,7000)

#add back first legend
ax.add_artist(leg1)

# Save to file.
fig.savefig('figs/Fig1_left.pdf')
plt.show()



