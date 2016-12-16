import pickle
import logging as log

def load_classes(filename):
    with open(filename, 'rb') as handle:
    par_config.classes = pickle.load(handle)
    log.info("Class Count: " + str(len(classes)))
    log.info("Classes: " + str(classes))
