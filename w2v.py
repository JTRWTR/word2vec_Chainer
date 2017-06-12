import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils

from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import collections

from chainer.utils import walker_alias

index2word = {}
word2index = {}
counts = collections.Counter()
dataset = []

with open('ptb.train.txt') as f:
    for line in f:
        for word in line.split():
            if word not in word2index:
                i = len(word2index)
                word2index[word]=i
                index2word[i]=word
            counts[word2index[word]] += 1
            dataset.append(word2index[word])

NUM_VOCAB = len(word2index)
DATASIZE = len(dataset)

print(NUM_VOCAB)
print( DATASIZE)

cs = [counts[w] for w in range(len(counts))]
power = np.float32(0.75)

p = np.array(cs,power.dtype)
sampler = walker_alias.WalkerAlias(p)

# print(sampler.sample(5))

class MyW2V( chainer.Chain):
    def __init__(self, n_vocab, n_units):
        super(MyW2V, self).__init__(
            embed = L.EmbedID(n_vocab, n_units)
        )
    def __call__(self, xb, yb, tb):
        xc = Variable( np.array(xb, dtype = np.int32))
        yc = Variable( np.array(yb, dtype = np.int32))
        tc = Variable( np.array(tb, dtype = np.int32))
        fv = self.fwd(xc, yc)
        return F.sigmoid_cross_entropy(fv, tc)
    def fwd(self, x, y):
        x1 = self.embed(x)
        x2 = self.embed(y)
        return F.sum(x1 * x2, axis = 1)

WINDOWSIZE = 3
NEGATIVE_SAMPLE_SIZE = 5

def mkbatset(dataset, ids):
    xb, yb, tb = [],[],[]
    for pos in ids:
        xid = dataset[pos]
        for i in range(1, WINDOWSIZE):
            p = pos - i
            if p >= 0:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(NEGATIVE_SAMPLE_SIZE):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
            p = pos + i
            if p <  DATASIZE:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(NEGATIVE_SAMPLE_SIZE):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
    return [xb, yb, tb]


demb = 100
model = MyW2V(NUM_VOCAB, demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

bs = 100
for epoch in range(10):
    print('epoch: {0}'.format(epoch))
    indexes = np.random.permutation(DATASIZE)
    for pos in range(0,  DATASIZE, bs):
        print(epoch, pos)
        ids = indexes[pos:(pos+bs) if (pos+bs) <  DATASIZE else  DATASIZE]
        xb, yb, tb = mkbatset(dataset, ids)
        model.zerograds()
        loss = model(xb, yb, tb)
        loss.backward()
        optimizer.update()

with open('w2v.model', 'w') as f:
    f.write('%d %d\n' % (len(index2word), 100))
    w = model.embed.W.data
    for i in range(w.shape[0]):
        v = ' '.join(['%f' % v for v in w[i]])
        f.write('%s %s\n' % (index2word[i], v))
