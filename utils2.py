from collections import OrderedDict
import os

def ParseLogFile(path):
    # read the log file
    filename = os.path.join(path, 'log.txt')

    file = open(filename)
    allLines = file.readlines()

    # split into all epochs
    epochIdces = []
    for ind, curLine in enumerate(allLines):
        if curLine[0:5] == 'Epoch':
            epochIdces.append(ind)
    cntEpochs = len(epochIdces)

    # read keywords
    keywords = []
    tempLine = allLines[epochIdces[0] + 1]
    keys = tempLine.split(',')
    for ind, entry in enumerate(keys):
        if ':' not in entry:
            continue

        split = entry.split(':')
        keywords.append(split[0])

    results = OrderedDict()
    for key in keywords:
        results[key.lstrip()] = []

    # read data from each epoch
    for curEpoch in range(0, cntEpochs - 1):
        curEpochLines = allLines[epochIdces[curEpoch] + 1:epochIdces[curEpoch + 1] - 1]

        for line in curEpochLines:
            info = line.split(',')
            for entry in info:
                if ':' not in entry:
                    continue
                split = entry.split(':')
                key = split[0].lstrip()

                results[key].append(float(split[1]))

    return results