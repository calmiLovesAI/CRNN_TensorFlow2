EPOCHS = 50
BATCH_SIZE = 8
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 100
IAMGE_CHANNELS = 3

save_model_dir = "saved_model/"
save_frequency = 5

test_picture_path = ""

# dataset
dataset_images = ""
dataset_label = ""
train_ratio = 0.6
valid_ratio = 0.2
train_label = ""
valid_label = ""
test_label = ""


char2index_map = {}

padding_value = len(char2index_map)