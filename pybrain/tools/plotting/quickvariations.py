__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain.utilities import subDict, dictCombinations
import pylab


def plotVariations(datalist, titles, genFun, varyperplot=None, prePlotFun=None, postPlotFun=None,
                   _differentiator=0.0, **optionlists):
    """ A tool for quickly generating a lot of variations of a plot.
    Generates a number of figures from a list of data (and titles).
    For each data item it produces one or more figures, each with one or more plots, while varying
    all options in optionlists (trying all combinations).
    :arg genFun: is the function that generates the curve to be plotted, for each set of options.
    :key varyperplot: determines which options are varied within a figure.
    :key prePlotFun: is called before the plots of a figure
    :key postPlotFun: is called after the plots of a figure (e.g. for realigning axes).
    """
    odl = subDict(optionlists, varyperplot, False)
    fdl = subDict(optionlists, varyperplot, True)
    # title contains file and non-varying parameters
    titadd1 = ''.join([k+'='+str(vs[0])[:min(5, len(str(vs[0])))]+' '
                    for k,vs in odl.items()
                    if len(vs) == 1])
    for x, tit in zip(datalist, titles):
        for figdict in sorted(dictCombinations(fdl.copy())):
            pylab.figure()

            # it also contains the parameters that don't vary per figure
            titadd2 = ''.join([k+'='+str(v)[:min(5, len(str(v)))]+' '
                               for k,v in figdict.items()])
            pylab.title(tit+'\n'+titadd1+titadd2)

            # code initializing the plot
            if prePlotFun is not None:
                prePlotFun(x)

            for i, odict in enumerate(sorted(dictCombinations(odl.copy()))):
                # concise labels
                lab = ''.join([k[:3]+'='+str(v)[:min(5, len(str(v)))]+'-'
                               for k,v in odict.items()
                               if len(odl[k]) > 1])
                if len(lab) > 0:
                    lab = lab[:-1]  # remove trailing '-'
                else:
                    lab = None
                generated = genFun(x, **dict(odict, **figdict))
                if generated is not None:
                    if len(generated) == 2:
                        xs, ys = generated
                    else:
                        ys = generated
                        xs = range(len(ys))
                    # the differentiator can slightly move the curves to be able to tell them apart if they overlap
                    if _differentiator != 0.0:
                        ys = generated+_differentiator*i

                    pylab.plot(xs, ys, label=lab)

            if postPlotFun is not None:
                postPlotFun(tit)
            # a legend is only necessary, if there are multiple plots
            if lab is not None:
                pylab.legend()
