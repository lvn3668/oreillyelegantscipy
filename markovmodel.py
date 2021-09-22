# Author: Lalitha Viswanathan
# Project: Interpolated Markov Model (order 6) on prokaryotic genomes
import string
import numpy as np
import toolz as tz
from matplotlib import pyplot as plt, axis
from toolz import curried as curry
from glob import glob
import itertools as itertoolz


#################################################################

def buildbasedictionaryandnmerdictionaries(alphabets: str) -> tuple[dict, dict]:
    """

    :return:
    :return:
    :rtype: object
    :param alphabets:
    :return:
    """
    basesdictionary: dict[string, int] = dict(zip(alphabets, range(4)))
    nmerhash: dict[object, object] = {
        2: {(base1, base2): (basesdictionary[base1], basesdictionary[base2]) for base1, base2 in
            itertoolz.product(basesdictionary, basesdictionary)},
        3: {(base1, base2, base3): (basesdictionary[base1], basesdictionary[base2], basesdictionary[base3]) for
            base1, base2, base3 in
            itertoolz.product(basesdictionary, basesdictionary, basesdictionary)},
        4: {(base1, base2, base3, base4): (
        basesdictionary[base1], basesdictionary[base2], basesdictionary[base3], basesdictionary[base4]) for
            base1, base2, base3, base4 in
            itertoolz.product(basesdictionary, basesdictionary, basesdictionary, basesdictionary)},
        5: {(base1, base2, base3, base4, base5): (
        basesdictionary[base1], basesdictionary[base2], basesdictionary[base3], basesdictionary[base4],
        basesdictionary[base5]) for
            base1, base2, base3, base4, base5 in
            itertoolz.product(basesdictionary, basesdictionary, basesdictionary, basesdictionary, basesdictionary)},
        6: {
            (base1, base2, base3, base4, base5, base6): (
            basesdictionary[base1], basesdictionary[base2], basesdictionary[base3], basesdictionary[base4],
            basesdictionary[base5], basesdictionary[base6]) for
            base1, base2, base3, base4, base5, base6 in
            itertoolz.product(basesdictionary, basesdictionary, basesdictionary, basesdictionary, basesdictionary,
                              basesdictionary)}}
    return basesdictionary, nmerhash


#################################################################
@tz.curry
def markov(seq: str, order: int, PDICThash: dict):
    numberofuniquechars = 4
    if order == 2:
        model = np.zeros((numberofuniquechars, numberofuniquechars))
    elif order == 3:
        model = np.zeros((numberofuniquechars, numberofuniquechars, numberofuniquechars))
    elif order == 4:
        model = np.zeros((numberofuniquechars, numberofuniquechars, numberofuniquechars, numberofuniquechars))
    elif order == 5:
        model = np.zeros(
            (numberofuniquechars, numberofuniquechars, numberofuniquechars, numberofuniquechars, numberofuniquechars))
    elif order == 6:
        model = np.zeros(
            (numberofuniquechars, numberofuniquechars, numberofuniquechars, numberofuniquechars, numberofuniquechars,
             numberofuniquechars))

    tz.last(tz.pipe(seq, curry.sliding_window(order),
                    curry.map(PDICThash.__getitem__),
                    curry.map(increment_model(model))))
    np.seterr(invalid='ignore')
    model /= np.sum(model, axis=1)[:, np.newaxis]
    return model


#################################################################
def is_sequence(line: str):
    return not line.startswith('>')


#################################################################
@tz.curry
def is_nucleotide(letter: chr, LDICTparam: dict):
    return letter in LDICTparam


#################################################################
@tz.curry
def increment_model(model: np.zeros, index: int):
    model[index] += 1

#################################################################
@tz.curry
def calculateimmprob(model, index, markovmodel2, markovmodel3, markovmodel4, markovmodel5, markovmodel6):
    model[index] +=markovmodel2+markovmodel3+markovmodel4+markovmodel5+markovmodel6

#################################################################
@tz.curry
def genome(file_pattern: str, LDICTparam: dict):
    """

    :param file_pattern: str
    :type LDICTparam: dict
    """
    return tz.pipe(file_pattern, glob, sorted, curry.map(open), tz.concat, curry.filter(is_sequence), tz.concat,
                   curry.filter(is_nucleotide(LDICTparam=LDICTparam)))


#################################################################
@tz.curry
def immtestseq(seq: str, order: int, markovmodel2: dict, markovmodel3: dict, markovmodel4: dict, markovmodel5: dict, markovmodel6: dict):
    model = np.zeros((4,4))
    tz.last(tz.pipe(seq, curry.sliding_window(order),
                    curry.map(markovmodel2.__getitem__(seq)),
                    curry.map(markovmodel3.__getitem__(seq)),
                    curry.map(markovmodel4.__getitem__(seq)),
                    curry.map(markovmodel5.__getitem__(seq)),
                    curry.map(markovmodel6.__getitem__(seq)),
                    curry.map(calculateimmprob(model))))
    np.seterr(invalid='ignore')
    model /= np.sum(model, axis=1)[:, np.newaxis]
#################################################################
def read1k(f):
    return f.read(1024)
#################################################################
def plot_model(model, labels, figure=None):
    fig = figure or plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(model, cmap='magma')
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=axcolor)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(range(4))
        axis.set_ticks_position('none')
        axis.set_ticklabels(labels)
    plt.show()
    return ax


#################################################################
if __name__ == '__main__':
    drosophilamelanogaster: str = "C:\\Users\\visu4\\Documents\\wcgna\\data\\SRP140558\\dm6.fa"
    alphabetinfastafile: str = "atgc"
    lenalphabets = len(alphabetinfastafile)
    basedict, Nmerdicts = buildbasedictionaryandnmerdictionaries(alphabetinfastafile)
    print(Nmerdicts.keys())
    print(Nmerdicts[2])

    # I.e. pipe(data, f, g, h) is equivalent to h(g(f(data)))
    markovmodel2: dict = tz.pipe(drosophilamelanogaster.lower(), genome(LDICTparam=basedict), curry.take(10 ** 7),
                                markov(order=2, PDICThash=Nmerdicts[2]))
    print(markovmodel2)


    markovmodel3: dict = tz.pipe(drosophilamelanogaster.lower(), genome(LDICTparam=basedict), curry.take(10 ** 7),
                                markov(order=3, PDICThash=Nmerdicts[3]))
    print("Order 3")
    print(markovmodel3)
    ax = plot_model(markovmodel3, labels='atgc')
    markovmodel4: dict = tz.pipe(drosophilamelanogaster.lower(), genome(LDICTparam=basedict), curry.take(10 ** 7),
                                markov(order=4, PDICThash=Nmerdicts[4]))
    print("Order 4")
    print(markovmodel4)

    markovmodel5: dict = tz.pipe(drosophilamelanogaster.lower(), genome(LDICTparam=basedict), curry.take(10 ** 7),
                                markov(order=5, PDICThash=Nmerdicts[5]))
    print("Order 5")
    print(markovmodel5)

    markovmodel6: dict = tz.pipe(drosophilamelanogaster.lower(), genome(LDICTparam=basedict), curry.take(10 ** 7),
                                markov(order=6, PDICThash=Nmerdicts[6]))
    print("Order 6")
    print(markovmodel6)


    #tz.pipe(drosophilamelanogaster.lower(), genome(LDICTparam=basedict), curry.take(10**7), immtestseq(6, markovmodel2, markovmodel3, markovmodel4, markovmodel5, markovmodel6))