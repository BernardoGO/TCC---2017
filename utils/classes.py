import pickle

def get_classes(filename):
    with open(filename, 'rb') as handle:
    classes = pickle.load(handle)
    log.info("Class Count: " + str(len(classes)))
    log.info("Classes: " + str(classes))
