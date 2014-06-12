import os
import sys
import numpy
import tables
import math
import gzip


def main():
    indexstart = int(sys.argv[1])
    indexcurrent = int(sys.argv[2])
    count = int(sys.argv[3])
    buf1 = 0
    fnames = [f.strip() for f in file('/data/lisa4/torabi/JeremieScripts/files.txt').readlines()]
    for num in range(indexcurrent,(indexstart+count)):																				
        buf1 = "%07d" %num
        path1 = '/data/lisa4/torabi/UCF101/Improvedfeature/' + fnames[num] + '.txt.gz'
        path2 = '/Tmp/zumerjer/UCF101/' + str(buf1) + '.hdf'
        # Store "x" in a chunked array with level 5 BLOSC compression...
        f = tables.openFile(path2, 'w')
        atom = tables.Float32Atom()
        filters = tables.Filters(complib='blosc', complevel=5)
        dataToInsert1 = numpy.loadtxt(gzip.open(path1))
        ds = f.createCArray(f.root, 'denseFeat', atom, dataToInsert1.shape,filters=filters)
        numFeat = len(dataToInsert1)
        numBin = math.floor(numFeat/20)
        dataToInsert=[]

        if numBin > 100:
            for i in range(0,20):
                start =  numBin*(i)
                if i< 19: 
                    end = numBin*(i+1)
                else:
                    end = numFeat+1 
                temp = dataToInsert1[start:end]
                numpy.random.shuffle(temp)
                dataToInsert1[start:end] = temp
            for m in range(0,int(numBin)):
                for n in range(0,20):
                    ind = int((n*numBin)+m)
                    dataToInsert.append(dataToInsert1[ind,:])
            lastInd =int(20*numBin)
            if numFeat-lastInd > 0:
                t= int(numFeat)
                for n1 in range(lastInd,t):				
                    dataToInsert.append(dataToInsert1[n1,:])	   
            ds[:] = dataToInsert
        else:
            numpy.random.shuffle(dataToInsert1)
            ds[:] = dataToInsert1    
           		
        f.flush()
        f.close()

if __name__ == "__main__":
	main()
