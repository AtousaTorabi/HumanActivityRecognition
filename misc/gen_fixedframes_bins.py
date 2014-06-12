import os
import sys
import numpy
import tables
import math
import gzip

import numpy as np




def convert(filelist, labels, num_trajs, indir, outdir,
            batch_size = 64, bin_size = 3):

    ### Idx save the bin idx associated with each video in a hdf files
    bins_idx = np.zeros((len(filelist), 2))
    ### Idx, used to save the traj idx associated with each bins in a hdf files
    traj_idx = np.zeros((0, 2))



    batch = 0
    batch_tot = len(filelist) / batch_size
    cur = 0
    max_bins = 0
    for batch in xrange(0, batch_tot):

        hdfpath = outdir + '/' +  "%07d" % batch + '.hdf'
        print(hdfpath)
        print batch, batch_tot, cur, 0

        # Store "x" in a chunked array with level 5 BLOSC compression...
        rows = np.sum(num_trajs[batch*batch_size:(batch+1)*batch_size])
        cols = 436
        print(rows, cols)
        f = tables.openFile(hdfpath, 'w')
        atom = tables.Float32Atom()
        filters = tables.Filters(complib='blosc', complevel=5)
        ds = f.createCArray(f.root, 'denseFeat', atom, (rows, cols),
                            filters = filters)

        ### Previous bins position
        prev = 0
        prev_traj=0
        for ba in xrange(batch * batch_size, (batch+1) * batch_size):
            ### Read feats
            path1 = indir + '/' + filelist[cur].strip()
            print(ba, path1)
            data = numpy.loadtxt(path1)

            print(num_trajs[cur], data.shape[0])
            assert(num_trajs[cur] == data.shape[0])

            ### Compute bins
            start_frame = data[0, 0]
            end_frame = data[-1, 0]
            nb_bins = int(np.ceil((end_frame - start_frame) / bin_size))
            if(nb_bins > max_bins):
                max_bins = nb_bins

            ### Update bins_idx
            bins_idx[cur][0] = prev
            bins_idx[cur][1] = prev + nb_bins
            prev = prev + nb_bins

            ### Previous traj position
            traj_pos = 0
            cur_traj_idx = np.zeros((nb_bins, 2))
            for b in xrange(0, nb_bins):

                #### Get the trajectory of the current bins
                bin_start_frame = b * bin_size + start_frame
                bin_end_frame = (b + 1) * bin_size + start_frame
                s = traj_pos
                while ((traj_pos < data.shape[0]) and
                       (data[traj_pos, 0] <= bin_end_frame)):
                    traj_pos = traj_pos + 1
                num_traj = traj_pos - s


                ### Compute cur_traj_index for bin
                cur_traj_idx[b][0] = prev_traj
                cur_traj_idx[b][1] = prev_traj + num_traj




                ### Randomize trajectory in each bins

                num_rngbin = math.floor(num_traj / 20)
                if num_rngbin > 100:
                    dataToInsert = []
                    for i in range(0, 20):
                        start =  num_rngbin * i
                        if i < 19:
                            end = num_rngbin * (i + 1)
                        else:
                            end = num_traj + 1
                        temp = data[s + start:s + end, :]
                        numpy.random.shuffle(temp)
                        data[s + start:s + end, :] = temp
                    for m in range(0, int(num_rngbin)):
                        for n in range(0,20):
                            ind = int((n * num_rngbin) + m)
                            dataToInsert.append(data[ind,:])
                    lastInd =int(20 * num_rngbin)
                    if num_traj - lastInd > 0:
                        t = int(num_traj)
                        for n1 in range(lastInd, t):
                            dataToInsert.append(data[n1,:])
                    ds[cur_traj_idx[b][0]:cur_traj_idx[b][1], :] = dataToInsert
                else:
                    dataToInsert = data[s:traj_pos, :]
                    numpy.random.shuffle(dataToInsert)
                    ds[cur_traj_idx[b][0]:cur_traj_idx[b][1], :] = dataToInsert[:, :]
                prev_traj = cur_traj_idx[b][1]



            # Update the number of video processed
            cur = cur + 1
            traj_idx = np.vstack([traj_idx, cur_traj_idx])

        ### Close files
        f.flush()
        f.close()

    ### Save idx
    np.savetxt(outdir + "/bins_idx.txt", bins_idx)
    np.savetxt(outdir + "/trajs_idx.txt", traj_idx)
    print max_bins


def read_list(filename):
    l = []
    with open(filename) as fp:
        for line in fp:
            l.append(line)

    return l

if __name__ == "__main__":

    if (len(sys.argv) != 6):
        print("%s: inputs_list labels num_trajs indir outdir" % sys.argv[0])
        exit(1)


    ## Important Parameter
    batch_size = 64
    bin_size = 3

    # Reads inputs
    l = read_list(sys.argv[1])
    labels = np.loadtxt(sys.argv[2])
    num_traj = np.loadtxt(sys.argv[3])

    ## Ensure that the number of samples is divisble by batch_size
    if (labels.shape[0] % batch_size != 0):
        toadd = batch_size - (labels.shape[0] % batch_size)
        for i in xrange(0, toadd):
            l.append(l[i])
        labels = np.append(labels, labels[0:toadd, :], axis=0)
        num_traj = np.append(num_traj, num_traj[0:toadd])

    ## Shuffle example
    idx = np.arange(0, len(l))
    np.random.shuffle(idx) #XXX: WITH SHUFFLING!
    l2 = l
    lab2 = labels
    num2 = num_traj
    for i in xrange(0, len(l)):
        l2[i]      = l[idx[i]]
        lab2[i, :] = labels[idx[i], :]
        num2[i]    = num_traj[idx[i]]
    l = l2
    num_traj = num2
    labels = lab2


    ## Save the traj files
    convert(l, labels, num_traj, sys.argv[4], sys.argv[5], batch_size, bin_size)
    ## Save the labels_file in a good location as well
    np.savetxt(sys.argv[5] + "/labels.txt", labels)
    np.savetxt(sys.argv[5] + "/rng_idx.txt", idx)
    np.savetxt(sys.argv[6] + '/files_lst.txt', numpy.array(l))
