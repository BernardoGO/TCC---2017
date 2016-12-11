import pickle

def get_classes(filename):
    with open(filename, 'rb') as handle:
    classes = pickle.load(handle)
    print(len(classes))
    print(classes)
