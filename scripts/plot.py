import yaml
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy

def get_values(d):
    h = {}
    machs, filters, ticks = [], [], []
    for k_mach, v_mach in d.iteritems():
        h[k_mach] = {}
        if not k_mach in machs:
            machs.append(k_mach)
        for k_filter, v_filter in v_mach.iteritems():
            for x, y in v_filter.iteritems():
                if not (k_filter in h[k_mach]):
                    h[k_mach][k_filter] = []
                if not k_filter in filters:
                    filters.append(k_filter)
                h[k_mach][k_filter].append(float(y))
    return (machs, filters, ticks, h)

def color(i):
    colors = "rgbymc"
    i = i % len(colors)
    return colors[i]

def set_graphic_properties(xlabel, ylabel, title, legend, ticks):
    majorLocator   = MultipleLocator(1)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(.2)

    ax = plt.subplot(111)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_view_interval(3,6)
    # plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axes().autoscale_view(False)
    plt.legend(tuple(legend),'upper right', fancybox=True)
    plt.grid(True)


f = open('data.yml')
d = yaml.load(f.read())
f.close

machs, filters, ticks, values = get_values(d)
mainmach = 'Prd3c'

for i, f in enumerate(filters):
    v = values[mainmach][f]
    arr = numpy.array(v)
    x = numpy.arange(1, len(arr) + 1)
    s = arr.sum()
    arr = 100 * arr / s
    plt.plot(x, arr, color(i) + '-o')


set_graphic_properties("CPUs", "Time % (percentage)", "Filters comparrison", filters, ticks)
plt.savefig('filters.png', dpi=192)
plt.cla()

for f in filters:
    for i, m in enumerate(machs):
        v = values[m][f]
        arr = numpy.array(v)
        x = numpy.arange(1, len(arr) + 1)
        plt.plot(x, arr, color(i) + '-o')

    set_graphic_properties("CPUs", "Time (milliseconds)", "Machines comparrison - " + f + " filter", machs, ticks)
    plt.savefig(f + '.png', dpi=192)
    plt.cla()

