import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
ddi = pd.read_csv('../data/bio-decagon-combo.csv')
drugs = list(set(ddi['STITCH 1'].tolist() + ddi['STITCH 2'].tolist()))
edges = list(set(ddi['Polypharmacy Side Effect']))
edic = {e:i for i, e in enumerate(edges)}
nd = len(drugs)
ne = len(edges)
print(nd)
edge_co = np.zeros((ne,ne))
drug_co = defaultdict(set)
for i, row in tqdm(ddi.iterrows(), total=len(ddi)):
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
