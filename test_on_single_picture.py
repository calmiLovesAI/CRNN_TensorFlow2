import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, IAMGE_CHANNELS, save_model_dir, char2index_map, test_picture_path
from core.crnn import CRNN
from core.predict import predict_text


def get_idx2char_map():
    idx2char = {}
    for k, v in char2index_map.items():
        idx2char[v] = k
    idx2char[-1] = "*"
    return idx2char


def get_final_output_string(output):
    output_tensor = tf.squeeze(output)
    idx2char = get_idx2char_map()
    output_string_list = []
    for i in range(output_tensor.shape[0]):
        output_string_list.append(idx2char[output_tensor[i].numpy()])
    output_string = "".join(output_string_list)
    return output_string


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # read image
    image_raw = tf.io.read_file(test_picture_path)
    image_tensor = tf.io.decode_image(contents=image_raw, channels=IAMGE_CHANNELS, dtype=tf.dtypes.float32)
    image_tensor = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image_tensor = tf.expand_dims(input=image_tensor, axis=0)

    # load model
    crnn_model = CRNN()
    crnn_model.load_weights(filepath=save_model_dir)

    pred = crnn_model(image_tensor, training=False)
    predicted_string = get_final_output_string(predict_text(pred))
    print(predicted_string)