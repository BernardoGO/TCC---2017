import pickle
import logging as log
import par_config

def load_classes(filename=par_config.classes_pickle_file):
    with open(filename, 'rb') as handle:
        par_config.classes = pickle.load(handle)
        log.info("Class Count: " + str(len(classes)))
        log.info("Classes: " + str(classes))
