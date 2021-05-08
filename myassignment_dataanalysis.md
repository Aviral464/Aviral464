import random # library for random number generation
import numpy as np # library for vectorized computation
import pandas as pd # library to process data as dataframes

import matplotlib.pyplot as plt # plotting library
# backend for rendering plots within the browser
%matplotlib inline 

from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs

print('Libraries imported.')

# data
x1 = [-4.9, -3.5, 0, -4.5, -3, -1, -1.2, -4.5, -1.5, -4.5, -1, -2, -2.5, -2, -1.5, 4, 1.8, 2, 2.5, 3, 4, 2.25, 1, 0, 1, 2.5, 5, 2.8, 2, 2]
x2 = [-3.5, -4, -3.5, -3, -2.9, -3, -2.6, -2.1, 0, -0.5, -0.8, -0.8, -1.5, -1.75, -1.75, 0, 0.8, 0.9, 1, 1, 1, 1.75, 2, 2.5, 2.5, 2.5, 2.5, 3, 6, 6.5]

print('Datapoints defined!')

colors_map = np.array(['b', 'r'])
def assign_members(x1, x2, centers):
    compare_to_first_center = np.sqrt(np.square(np.array(x1) - centers[0][0]) + np.square(np.array(x2) - centers[0][1]))
    compare_to_second_center = np.sqrt(np.square(np.array(x1) - centers[1][0]) + np.square(np.array(x2) - centers[1][1]))
    class_of_points = compare_to_first_center > compare_to_second_center
    colors = colors_map[class_of_points + 1 - 1]
    return colors, class_of_points

print('assign_members function defined!')


# update means
def update_centers(x1, x2, class_of_points):
    center1 = [np.mean(np.array(x1)[~class_of_points]), np.mean(np.array(x2)[~class_of_points])]
    center2 = [np.mean(np.array(x1)[class_of_points]), np.mean(np.array(x2)[class_of_points])]
    return [center1, center2]

print('assign_members function defined!')


def plot_points(centroids=None, colors='g', figure_title=None):
    # plot the figure
    fig = plt.figure(figsize=(15, 10))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)
    
    centroid_colors = ['bx', 'rx']
    if centroids:
        for (i, centroid) in enumerate(centroids):
            ax.plot(centroid[0], centroid[1], centroid_colors[i], markeredgewidth=5, markersize=20)
    plt.scatter(x1, x2, s=500, c=colors)
    
    # define the ticks
    xticks = np.linspace(-6, 8, 15, endpoint=True)
    yticks = np.linspace(-6, 6, 13, endpoint=True)

    # fix the horizontal axis
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # add tick labels
    xlabels = xticks
    ax.set_xticklabels(xlabels)
    ylabels = yticks
    ax.set_yticklabels(ylabels)

    # style the ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params('both', length=2, width=1, which='major', labelsize=15)
    
    # add labels to axes
    ax.set_xlabel('x1', fontsize=20)
    ax.set_ylabel('x2', fontsize=20)
    
    # add title to figure
    ax.set_title(figure_title, fontsize=24)

    plt.show()

print('plot_points function defined!')

plot_points(figure_title='Scatter Plot of x2 vs x1')
"text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "image/png": {
       "height": 267,
       "width": 390
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# QQ Plot\n",
    "from numpy.random import seed\n",
    "from numpy.random import randn\n",
    "from statsmodels.graphics.gofplots import qqplot\n",
    "from matplotlib import pyplot\n",
    "# seed the random number generator\n",
    "seed(1)\n",
    "# q-q plot\n",
    "qqplot(data, line='s')\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are some large gaps, especially at the top of the graph, which is to be expected since our variables are not strongly correlated."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Statistics=0.895, p=0.000\n",
      "Sample does not look Gaussian (reject H0)\n"
     ]
    }
   ],
   "source": [
    "# Shapiro-Wilk Test\n",
    "from numpy.random import seed\n",
    "from numpy.random import randn\n",
    "from scipy.stats import shapiro\n",
    "# seed the random number generator\n",
    "seed(1)\n",
    "# normality test\n",
    "stat, p = shapiro(data)\n",
    "print('Statistics=%.3f, p=%.3f' % (stat, p))\n",
    "# interpret\n",
    "alpha = 0.05\n",
    "if p > alpha:\n",
    "    print('Sample looks Gaussian (fail to reject H0)')\n",
    "else:\n",
    "    print('Sample does not look Gaussian (reject H0)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "D'Agostino's K² test allowed us to calculate summary statistics from our data, namely kurtosis and skewness, to determine if the distribution of the data deviates from the normal distribution, named after Ralph D'Agostino."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Statistics=0.102, p=0.950\n",
      "Sample looks Gaussian (fail to reject H0)\n"
     ]
    }
   ],
   "source": [
    "# D'Agostino and Pearson's Test\n",
    "from numpy.random import seed\n",
    "from numpy.random import randn\n",
    "from scipy.stats import normaltest\n",
    "# seed the random number generator\n",
    "seed(1)\n",
    "# normality test\n",
    "stat, p = normaltest(data)\n",
    "print('Statistics=%.3f, p=%.3f' % (stat, p))\n",
    "# interpret\n",
    "alpha = 0.05\n",
    "if p > alpha:\n",
    "    print('Sample looks Gaussian (fail to reject H0)')\n",
    "else:\n",
    "    print('Sample does not look Gaussian (reject H0)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Anderson-Darling Test is a statistical test that can be used to assess whether a data sample comes from one of many known data samples, named after Theodore Anderson and Donald Darling.\n",
    "\n",
    "It can be used to test whether a data sample is normal. The test is a modified version of a more sophisticated non-parametric goodness-of-fit statistical test called the Kolmogorov-Smirnov test.\n",
    "\n",
    "A feature of the Anderson-Darling test is that it returns a list of critical values rather than a single p-value."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Statistic: 0.220\n",
      "1.000: 1.053, data looks normal (fail to reject H0)\n"
     ]
    }
   ],
   "source": [
    "# Anderson-Darling Test\n",
    "from numpy.random import seed\n",
    "from numpy.random import randn\n",
    "from scipy.stats import anderson\n",
    "# seed the random number generator\n",
    "seed(1)\n",
    "# normality test\n",
    "result = anderson(data)\n",
    "print('Statistic: %.3f' % result.statistic)\n",
    "p = 0\n",
    "for i in range(len(result.critical_values)):\n",
    "    sl, cv = result.significance_level[i], result.critical_values[i]\n",
    "if result.statistic < result.critical_values[i]:\n",
    "    print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))\n",
    "else:\n",
    "    print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can interpret the results by not rejecting the null hypothesis that the data are normal if the calculated test statistic is below the critical value at a chosen significance level."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
