import pickle
with open('path3.pkl', 'rb') as f:
    data = pickle.load(f)
with open('paths.txt', 'w') as f:
    for path in data:
        f.write('->'.join(['(%d,%s)'%x for x in path]) + '\n')

