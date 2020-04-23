import pickle
from annoy import AnnoyIndex
import numpy as np
import tqdm
with open('edge_cooccurence.pkl', 'rb') as f:
    data = pickle.load(f)

ne = data['edic'].shape[0]
t = AnnoyIndex(ne, 'angular')
i = 0
vecs = []
for key, edges in tqdm.tqdm(data['drud_dic'].items()):
    vec = np.zeros(ne)
    vec[list(edges)] = 1
    t.add_item(i,vec)
    vecs.append(vec)
    i+=1

with open('vectors.pkl', 'wb') as f:
    pickle.dump(np.stack(vecs), f)
t.build(10)
t.save('text.annoy')
