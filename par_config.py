img_width, img_height = 500, 500
sizex, sizey = 1936, 1296#3024,4032
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 150000
nb_validation_samples = 24000
epochs = 60
batch_size = 16
imagePath = "collection/Images/users/his109/grier_food_images/"


dataset_csv_file = "supervis.csv"
dataset_csv_path = ""
classes_pickle_file = 'classesROI.pickle'

_CLASSES_TO_IGNORE_ = ["plate", "rag"]

test_images_folder = "test_images/"
test_images_xml_folder = "test_images/"
test_image_name = "dsc_1601"

images_extension = ".jpg"
par_config.STRIDES = 1
