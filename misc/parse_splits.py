import numpy
import os, os.path

lsts = []
output = []
num_classes = 101
prefix = '/data/lisa4/torabi/JeremieScripts/'

lsts.append(file(prefix + 'list.txt'))
for l in ('files', 'idx', 'labels'):
    output.append(file(prefix + l + '.txt', mode='w'))

lsts = [y.strip() for y in lsts[0].readlines()]
idx = range(len(lsts))

label_names = file(prefix + 'classInd.txt').readlines()
label_names = [x.split(' ') for x in label_names]
label_nums = [x for (x, y) in label_names]
label_txt = [y.strip() for (x, y) in label_names]

labels = []
for l in range(len(lsts)):
    labels.append(label_txt.index(os.path.split(lsts[l])[0]))


def onehot(x):
    y = numpy.zeros(num_classes)
    y[x] = 1.
    return y

labels = numpy.array([onehot(x) for x in labels])

for l in range(10*(len(lsts))):
    rnd1 = numpy.random.randint(0, len(lsts))
    rnd2 = numpy.random.randint(0, len(lsts))
    tmp = idx[rnd1]
    idx[rnd1] = idx[rnd2]
    idx[rnd2] = tmp

lsts = numpy.array(lsts)[numpy.array(idx)]
labels = numpy.array(labels)[numpy.array(idx)]

for k in lsts:
    output[0].write(k + '\n')
for k in idx:
    output[1].write(str(k) + '\n')
for k in labels:
    for l in k:
        output[2].write(str(l) + ' ')
    output[2].write('\n')
