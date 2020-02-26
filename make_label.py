import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, IAMGE_CHANNELS, dataset_images, char2index_map, padding_value


class Label(object):
    def __init__(self):
        super(Label, self).__init__()
        self.image_root = dataset_images
        self.padding_length = IMAGE_WIDTH // 4

    def make_label(self, batch_data):
        images = []
        labels = []
        for i in range(batch_data.shape[0]):
            image_name_and_label = bytes.decode(batch_data[i].numpy(), encoding="utf-8")
            image_name_and_label = image_name_and_label.strip().split(", ")
            image_name, image_label = image_name_and_label[0], image_name_and_label[1]
            images.append(self.read_image(image_name))
            labels.append(self.sequence_padding(self.read_label(image_label)))
        images_tensor = tf.stack(values=images, axis=0)
        labels_tensor = tf.convert_to_tensor(value=labels, dtype=tf.dtypes.int32)
        return images_tensor, labels_tensor

    def sequence_padding(self, label_list):
        if len(label_list) > self.padding_length:
            raise ValueError("The length of the label must be less than or equal to the padding length.")
        else:
            while len(label_list) < self.padding_length:
                label_list.append(padding_value)
        return label_list

    def read_image(self, image_name):
        image_path = self.image_root + image_name
        image_raw = tf.io.read_file(image_path)
        image_tensor = tf.io.decode_image(contents=image_raw, channels=IAMGE_CHANNELS, dtype=tf.dtypes.float32)
        image_tensor = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        return image_tensor

    def read_label(self, image_label):
        label = []
        for i in range(len(image_label)):
            label.append(char2index_map[image_label[i]])
        return label