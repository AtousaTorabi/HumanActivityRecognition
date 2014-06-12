import tables
import sys


def merge(out, fnames):
    data = tables.openFile(out, mode='a')

    for fname in fnames:
        f = tables.openFile(fname, mode='r')
        raw_targets = f.root.denseFeat
        
        if 'denseFeat' in data.root:
            prev_data = data.root.denseFeat
            targets = data.createCArray(data.root, '_y', atom=tables.Float32Atom(), shape=((raw_targets.shape[0]+prev_data.shape[0],436)))
            targets[:prev_data.shape[0],:] = prev_data[:,:]
            targets[prev_data.shape[0]:,:] = raw_targets[:,:]
            data.flush()
            data.removeNode(data.root, "denseFeat", 1)
        else:
            targets = data.createCArray(data.root, '_y', atom=tables.Float32Atom(), shape=((raw_targets.shape[0],436)))
            targets[:,:] = raw_targets[:,:]
            data.flush()


        data.renameNode(data.root, "denseFeat", "_y")
        data.flush()

        f.close()

    data.close()

if __name__ == '__main__':
    merge(sys.argv[1], sys.argv[2:])
