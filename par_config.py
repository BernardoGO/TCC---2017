img_width, img_height = 500, 500
sizex, sizey = 1936, 1296#3024,4032

retrain_epochs = 200
retrain_batch_size = 32
imagePath = "collection/Images/users/his109/grier_food_images/"


dataset_csv_file = "supervis.csv"
dataset_csv_path = ""
classes_pickle_file = 'classesROI.pickle'

_CLASSES_TO_IGNORE_ = ["plate", "rag"]

test_images_folder = "test_images/"
test_images_xml_folder = "test_images/"
test_image_name = "dsc_1601"

images_extension = ".jpg"
STRIDES = 1

pretrained_weights_file = "image_500.h5"

joinRoIs_considered_coverage = 0.001
final_optimizer = "rmsprop"

font_filename = "sans-serif.ttf"

ignore_notJoined_boxes = True

classify_batchsize = 64

final_weights_filename = 'image_500_retr.h5'
final_model_filename = "model_500_retr.h5"
final_class_count = 25
tensorboard_file = './Graph'
