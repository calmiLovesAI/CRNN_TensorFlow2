import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, IAMGE_CHANNELS, dataset_images, char2index_map, padding_value
from core.sparse_tensor import GenerateSparseTensor


class Label:
    def __init__(self):
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
            labels.append(self.read_label(image_label))
        images_tensor = tf.stack(values=images, axis=0)
        label_sparse_tensor = self.sequence_to_sparse_tensor(label_list=labels)
        return images_tensor, label_sparse_tensor, labels

    def sequence_to_sparse_tensor(self, label_list):
        indices_list = []
        values_list = []
        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                indices_list.append([i, j])
                values_list.append(label_list[i][j])
        dense_shape_list = [len(label_list), self.padding_length]
        generate_sparse_tensor = GenerateSparseTensor()
        return generate_sparse_tensor(indices_list=indices_list,
                                      values_list=values_list,
                                      dtype=tf.dtypes.int32,
                                      dense_shape=dense_shape_list)

    def read_image(self, image_name):
        image_path = self.image_root + image_name
        image_raw = tf.io.read_file(image_path)
        image_tensor = tf.io.decode_image(contents=image_raw, channels=IAMGE_CHANNELS, dtype=tf.dtypes.float32)
        image_tensor = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        return image_tensor

    @staticmethod
    def read_label(image_label):
        label = []
        for i in range(len(image_label)):
            label.append(char2index_map[image_label[i]])
        return label