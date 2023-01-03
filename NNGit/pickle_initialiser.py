

import dill as pickle

#data = []
#dump = open("data.txt", "wb")
#pickle.dump(data, dump)


def getfile():
    data = []
    dump = open("data.txt", "rb")
    file = pickle.load(dump)
    print(f"number of entries: {len(file)}")
    return file
    


