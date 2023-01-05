import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt

# colors
Red = "#f60101"
Moss = "#a49d6c"
Powder_Blue = "#a1d3d2"
Blue = "#0da9d4"
Dark = "#0b272d"
Green = "#188952"
color_list = [Blue, Green, Red, Moss, Powder_Blue, Dark]

# matplotlib modifications
fe = fm.FontEntry(
    fname='./Montserrat-Regular.ttf',
    name='Montserrat')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
sns.set(rc={
             'axes.axisbelow': False,
             'axes.edgecolor': 'lightgrey',
             'axes.facecolor': 'None',
             'axes.grid': False,
             'axes.labelcolor': 'dimgrey',
             'axes.spines.right': False,
             'axes.spines.top': False,
             'figure.facecolor': 'white',
             'lines.solid_capstyle': 'round',
             'patch.edgecolor': 'w',
             'patch.force_edgecolor': True,
             'text.color': 'dimgrey',
             'xtick.bottom': False,
             'xtick.color': 'dimgrey',
             'xtick.direction': 'out',
             'xtick.top': False,
             'ytick.color': 'dimgrey',
             'ytick.direction': 'out',
             'ytick.left': False,
             'ytick.right': False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})
mpl.rcParams['font.family'] = fe.name
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)