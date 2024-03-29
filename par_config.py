


sizex, sizey = 1936, 1296#3024,4032

img_width, img_height = 500,500


retrain_epochs = 5#200
retrain_batch_size = 32
xmlPath = "collection/Annotations/users/his109/grier_food_images/"
weights_folder = "weights/"
imagePath = "collection/Images/users/his109/grier_food_images/"
compare_RoIs = True

dataset_csv_file = "supervis.csv"
dataset_csv_path = ""
classes_pickle_file = weights_folder+'classesROI.pickle'

_CLASSES_TO_IGNORE_ = ["plate", "rag"]

test_images_folder = "test_images/"
test_images_xml_folder = "test_images/"
test_image_name = "dsc_1703"

images_extension = ".jpg"
STRIDES = 1

backgroundThreshold = 0.60

pretrained_weights_file =  weights_folder+"image_500.h5"

joinRoIs_considered_coverage = -0.05
final_optimizer = "rmsprop"

font_filename = "sans-serif.ttf"

ignore_notJoined_boxes = True

classify_batchsize = 64

consider_full_conv = True

final_weights_filename = weights_folder+'image_500_NONretr.h5'
final_model_filename =  weights_folder+"model_500_NONretr.h5"
final_class_count = 25
tensorboard_file = './Graph'