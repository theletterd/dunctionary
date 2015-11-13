from collections import namedtuple

import math
import numpy

CurveData = namedtuple('CurveData', ['mean_yes', 'mean_no', 'stddev_yes', 'stddev_no', 'gamma', 'beta', ])


def get_curve_params(yes_scores, no_scores):

    mean_yes = numpy.mean(yes_scores)
    mean_no = numpy.mean(no_scores)
    stddev_yes = numpy.std(yes_scores)
    stddev_no = numpy.std(no_scores)

    beta = ((mean_yes - mean_no) / (stddev_yes + stddev_no)) * stddev_no
    gamma = mean_no + beta

    return CurveData(
        mean_yes=mean_yes,
        mean_no=mean_no,
        stddev_yes=stddev_yes,
        stddev_no=stddev_no,
        beta=beta,
        gamma=gamma
    )


def probability(curvedata, svm_output, phi=2, psi=0.1):

    S = psi / curvedata.stddev_yes
    return (math.tanh(S * (svm_output - curvedata.gamma + (phi * curvedata.stddev_yes))) + 1) / 2
