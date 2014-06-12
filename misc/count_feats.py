import gzip
import numpy
import sys

#invoke: [] train 0 10 100 -> process train set's 1st split between incidces 0 and 10.
what = sys.argv[1]
which = ('train', 'test').index(what)
num = int(sys.argv[2])
start = (sys.argv[3])
end = (sys.argv[4])

files = file(what + str(num+1) + '_files.txt').readlines()
idx = [int(a.strip()) for a in file(what + str(num+1) + '_idx.txt').readlines()]
output = file('NumFeatures_' + what + str(num+1) + '_part_' + str(start) + '.txt', mode='w')

indices = [idx.index(k) for k in range(max(idx)+1)[int(start):int(end)]]
fnames = numpy.array(files)[indices]
#files = [gzip.open('Improvedfeature/' + f[:-1] + '.txt.gz') for f in fnames]
#NOTE: doesn't work: too many open files error
for fn in fnames:
    f = gzip.open('Improvedfeature/' + fn[:-1] + '.txt.gz')
    output.write(str(len(f.readlines())) + '\n')
    f.close()

output.close()
