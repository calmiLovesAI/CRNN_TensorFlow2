import tensorflow as tf
from configuration import BATCH_SIZE, dataset_images, dataset_label, \
    train_ratio, valid_ratio, train_label, valid_label, test_label


class Dataset:
    def __init__(self):
        self.images_dir = dataset_images
        self.label_dir = dataset_label

        self.train_label_dir = train_label
        self.valid_label_dir = valid_label
        self.test_label_dir = test_label

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        # 划分数据集

    def split_dataset(self):
        print("Splitting dataset...")
        with open(file=self.label_dir, mode="r") as f:
            image_label_list = f.readlines()
        while image_label_list:
            if image_label_list[-1] == "\n":
                image_label_list = image_label_list[:-1]
            else:
                break
        num_total_samples = len(image_label_list)
        num_train_samples = int(num_total_samples * self.train_ratio)
        num_valid_samples = int(num_total_samples * self.valid_ratio)
        train_list = image_label_list[:num_train_samples]
        valid_list = image_label_list[num_train_samples:num_train_samples + num_valid_samples]
        test_list = image_label_list[num_train_samples + num_valid_samples:]
        self.__write_to_txt(txt_file=self.train_label_dir, information_list=train_list)
        self.__write_to_txt(txt_file=self.valid_label_dir, information_list=valid_list)
        self.__write_to_txt(txt_file=self.test_label_dir, information_list=test_list)

    @staticmethod
    def __write_to_txt(txt_file, information_list):
        with open(file=txt_file, mode="a+") as f:
            for item in information_list:
                f.write(item)

    @staticmethod
    def __get_size(dataset):
        count = 0
        for _ in dataset:
            count += 1
        return count

    def __generate_dataset(self, label_file):
        dataset = tf.data.TextLineDataset(filenames=label_file)
        dataset_size = self.__get_size(dataset)
        dataset = dataset.batch(batch_size=BATCH_SIZE)
        return dataset, dataset_size

    def train_dataset(self):
        train_set, train_size = self.__generate_dataset(label_file=self.train_label_dir)
        return train_set, train_size

    def valid_dataset(self):
        valid_set, valid_size = self.__generate_dataset(label_file=self.valid_label_dir)
        return valid_set, valid_size

    def test_dataset(self):
        test_set, test_size = self.__generate_dataset(label_file=self.test_label_dir)
        return test_set, test_size
