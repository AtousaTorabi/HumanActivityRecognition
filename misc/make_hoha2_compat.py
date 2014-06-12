import numpy

data = numpy.loadtxt(file('/Tmp/torabi/ImprovedfeatureHDF/Tags.txt'))
out = file('/u/zumerjer/torabi/JeremieScripts/HOHA2Tags_correct.txt', mode='w')

hoha2_cols = [188, 108, 167, 260, 35, 42, 317, 77, 123, 12, 11, 18]

for d in data:
    new = numpy.zeros((12,))
    here = [hoha2_cols.index(x) for x in numpy.where(d==1)[0] if x in hoha2_cols]
    if here:
        new[numpy.array(here)] = 1.
    for a in new:
        out.write(str(a) + ' ')
    out.write('\n')
