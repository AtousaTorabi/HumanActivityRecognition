import pylearn2, theano, pickle, numpy
import atousaDatasetNew as adn
from pylearn2_ECCV2014.meanAvgPrec import meanAveragePrecisionTheano
import sys, time
import matplotlib.pyplot as plt

#XXX: hasn't been tested properly since last modified.

path = '/data/lisa4/torabi/ImprovedfeatureNewHDF/'
f2i = {50:0, 100:1, 500:2} #num of trajectories to index mapping
i2f = dict(zip(f2i.values(), f2i.keys())) #index to features
which = {'train':0, 'test':1} #name of set to index
what = dict(zip(which.values(), which.keys())) #index to name
batch_cnt = {'train':15, 'test':3} #how many files to the set
n_ensembles = [10] #how many samples to average in the ensemble
splits = {0:10000, 1:50000, 2:100000, 3:150000, 4:300000}
resample = 5

voting_mode = False
if len(sys.argv) > 1 and sys.argv[1] == '--vote':
    voting_mode = True #NOTE: voting mode has NOT been tested yet.

dset = [[0]*len(f2i), [0]*len(f2i)]
model = [0]*len(f2i)
data = [[(numpy.array([]),numpy.array([]))]*len(f2i), [(numpy.array([]),numpy.array([]))]*len(f2i)]

idx = [0,0]
idx[which['train']] = numpy.cast['int'](numpy.loadtxt(path+'train/Newtrain.txt'))
idx[which['test']] = numpy.cast['int'](numpy.loadtxt(path+'test/Newtest.txt'))
num_feats = numpy.cast['int'](numpy.loadtxt('NumFeatures.txt'))

x = theano.tensor.tensor4()
y_hat = theano.tensor.fmatrix()
y = theano.tensor.fmatrix()

predict_fn = [0]*len(f2i)

for i in range(len(model)):
    model[i] = pickle.load(file(path + 'latest2class3_' + str(i2f[i]) + '.pkl'))
    dset[which['train']][i] = adn.AtousaDataset('train', num_feats=i2f[i])
    dset[which['test']][i] = adn.AtousaDataset('test', num_feats=i2f[i])
    predict_fn[i] = theano.function([x], model[i].fprop(x))

theano_map = theano.function([y_hat, y], meanAveragePrecisionTheano(y_hat, y))

def predict(data, i):
    transform = lambda x: x
    if voting_mode:
        transform = lambda x: onehot(x)

    return transform(numpy.array(predict_fn[i](data)))

def mean_ap(target, pred):
    return numpy.array(theano_map(target, pred))

def onehot(pred):
    oh = numpy.zeros_like(pred)
    for i, j in enumerate(numpy.argmax(pred, axis=1)):
        oh[i][j] = 1
    return oh

results = []

for rs in range(resample):
    results.append([])
    for t in range(len(splits)+1):
        results[rs].append([])
        data = []
        labels = []
        a = max(t-1, 0)
        b = min(t, len(splits)-1)

        indices = []

        for j in what:
            results[rs][t].append([])
            data.append([])
            labels.append([])
            for i in range(len(model)):
                results[rs][t][j].append([])
                data[j].append([])
                labels[j].append(numpy.array([]))

                if b == 0:
                    indices = num_feats[idx[j]] < splits[a]
                elif a == b:
                    indices = num_feats[idx[j]] >= splits[b]
                else:
                    indices = (num_feats[idx[j]] < splits[b]) & (num_feats[idx[j]] >= splits[a])

                print 'Videos in this category:', sum(indices)

                timer = -time.time()
                localtime = time.localtime(-timer)
                print what[j] + str(i2f[i]) + '... %02d:%02d:%02d' % (localtime.tm_hour, localtime.tm_min, localtime.tm_sec)

                first_run = True
                for c in range(max(n_ensembles)):
                    dset_iter = dset[j][i].iterator(num_batches=batch_cnt[what[j]], targets=True)
                    data[j][i].append(numpy.array([]))

                    for k in range(batch_cnt[what[j]]):
                        new_batch = dset_iter.next()
                        the_slice = indices[k*new_batch[0].shape[-1]:(k+1)*new_batch[0].shape[-1]] #new_batch[0].shape[-1] should be 64
                        if first_run:
                            if not labels[j][i].size:
                                labels[j][i] = new_batch[1][the_slice]
                            else:
                                labels[j][i] = numpy.concatenate((labels[j][i], new_batch[1][the_slice]))

                        if (not the_slice.any()):
                            continue

                        if not data[j][i][c].size:
                            data[j][i][c] = predict(new_batch[0][:,:,:,the_slice], i)
                        else:
                            data[j][i][c] = numpy.concatenate((data[j][i][c], predict(new_batch[0][:,:,:,the_slice], i)))
                    first_run = False

                for nn, n in enumerate(n_ensembles):
                    if n == 1:
                        cut = numpy.array(data[j][i])
                    else:
                        cut = numpy.array(data[j][i][:n])

                    this_data = (numpy.sum(cut, axis=0)/len(cut), labels[j][i])
                    if this_data[1].size:
                        results[rs][t][j][i].append(mean_ap(this_data[1], this_data[0]))

                        print what[j], 'MAP -', str(i2f[i]),\
                            'trajectories model, videos with',\
                            str(splits[a]), 'or less' if b == 0 else 'or more' if a == b else 'to ' + str(splits[b]),\
                            'trajectories, and',\
                            str(n), 'trajectories samplings:',\
                            results[rs][t][j][i][nn]

                    else:
                        results[rs][t][j][i].append(numpy.array([0., 0., 0.]))
                        print what[j], 'from', str(splits[a]), 'to', str(splits[b]), 'trajectories is empty!'

#results order: rs; split; which set; which model; # in ensemble
bins = []
bins.extend([splits[i]-(splits[i-1] if i > 0 else 0) for i in splits])
bins.extend([max(num_feats)-bins[-1]])
results_train = numpy.array(results)[:,:,which['train']] #[resample x splits x [selected set] x model x n_ensembles] -> array of [minAP, meanAP, maxAP]
results_test = numpy.array(results)[:,:,which['test']]

x = range(len(model))
xnames = [str(i2f[y]) + " traj. model" for y in x]

for i in splits:
    for k in range(len(n_ensembles)):
        fig = plt.figure()
        plt.boxplot(results_train[:,i,:,k,1], sym='')
        plt.xlabel('Model')
        plt.ylabel('Mean average precision')
        plt.title('Ensemble performance with ' + str(n_ensembles[k]) + ' samplings (' + str(splits[i]) + ' or less trajectories, train set)')
        fig.get_axes()[0].set_xticklabels(xnames)
        plt.show()

for i in range(len(splits)):
    for k in range(len(n_ensembles)):
        fig = plt.figure()
        plt.boxplot(results_test[:,i,:,k,1], sym='')
        print results_test[:,i,:,k,1]
        plt.xlabel('Model')
        plt.ylabel('Mean average precision')
        plt.title('Ensemble performance with ' + str(n_ensembles[k]) + ' samplings (' + str(splits[i]) + ' trajectories, test set)')
        fig.get_axes()[0].set_xticklabels(xnames)
        plt.show()
