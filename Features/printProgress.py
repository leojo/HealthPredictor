#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
# Print iterations progress (just to make it pretty)
def printProgress (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, barLength = 50):
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()