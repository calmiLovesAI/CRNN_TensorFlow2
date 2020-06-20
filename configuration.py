

class Config:
    EPOCHS = 50
    BATCH_SIZE = 2
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 100
    IAMGE_CHANNELS = 3

    save_model_dir = "saved_model/"
    save_frequency = 10

    test_picture_path = ""

    # dataset
    dataset_images = ""
    dataset_label = ""
    train_ratio = 0.6
    valid_ratio = 0.2
    train_label = ""
    valid_label = ""
    test_label = ""

    charset_file = ""



    @classmethod
    def get_idx2char(cls):
        with open(file=cls.charset_file, mode="r", encoding="utf-8") as f:
            char_list = f.readlines()
        idx2char = dict((i, c.strip("\n")) for i, c in enumerate(char_list))
        idx2char[len(char_list)] = "BLANK"
        return idx2char

    @classmethod
    def get_char2idx(cls):
        idx2char = cls.get_idx2char()
        char2idx = dict((v, k) for k, v in idx2char.items())
        return char2idx


