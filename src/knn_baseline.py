import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pickle
from annoy import AnnoyIndex
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


ddi = pd.read_csv('../data/bio-decagon-combo.csv')
train, test = train_test_split(ddi, test_size=0.2)
drugs = list(set(ddi['STITCH 1'].tolist() + ddi['STITCH 2'].tolist()))
edges = sorted(list(set(ddi['Polypharmacy Side Effect'])))
edic = {e:i for i, e in enumerate(edges)}
nd = len(drugs)
ne = len(edges)
t = AnnoyIndex(ne, 'angular')
print(nd)
edge_co = np.zeros((ne,ne))
drug_co = defaultdict(set)
for i, row in tqdm(train.iterrows(), total=len(train)):
    d1 = row['STITCH 1']
    d2 = row['STITCH 2']
    e = row['Polypharmacy Side Effect']
    drug_co[d1,d2].add(edic[e])
    drug_co[d2,d1].add(edic[e])

for (d1,d2),es in tqdm(drug_co.items(), total=len(drug_co)):
    for e in es:
        for f in es:
            if e==f:
                continue
            edge_co[e,f] +=1
print(nd)
print(len(drug_co))


with open('edge_cooccurence.pkl', 'wb') as f:
    pickle.dump({'drug_dic':drug_co, 'edic': edge_co}, f)



def get_vec(edges, ne):
    vec = np.zeros(ne)
    vec[list(edges)] = 1
    return vec

t = AnnoyIndex(ne, 'angular')
i = 0
vecs = []
for key, edges in tqdm(drug_co.items()):
    vec = get_vec(edges, ne)
    t.add_item(i,vec)
    vecs.append(vec)
    i+=1
vecs = np.stack(vecs)
t.build(10)

prediction = []
target = []
K = 20
for i, row in tqdm(test.iterrows(), total=len(test)):
    d1 = row['STITCH 1']
    d2 = row['STITCH 2']
    e = row['Polypharmacy Side Effect']
    d3 = np.random.choice(drugs)
    vec = get_vec(drug_co[d1,d2], ne)
    nn = t.get_nns_by_vector(vec, K)
    prediction.append(np.mean(vecs[nn,edic[e]]))
    target.append(1)

    vec = get_vec(drug_co[d1,d3], ne)
    nn = t.get_nns_by_vector(vec, K)
    prediction.append(np.mean(vecs[nn,edic[e]]))
    target.append(0)

print(prediction)
print(np.mean(np.equal(prediction, target)))
print(roc_auc_score(target, prediction))
print(average_precision_score(target, prediction))
