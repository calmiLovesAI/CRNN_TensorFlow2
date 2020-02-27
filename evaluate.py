import tensorflow as tf
from core.read_dataset import Dataset
from core.make_label import Label
from core.crnn import CRNN
from core.loss import CTCLoss
from core.metric import Accuracy
from core.predict import predict_text
from configuration import save_model_dir


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # dataset
    dataset = Dataset()
    test_set, test_size = dataset.test_dataset()

    # model
    crnn_model = CRNN()
    crnn_model.load_weights(filepath=save_model_dir+"saved_model")

    # loss and metrics
    loss_object = CTCLoss()
    loss_metric = tf.metrics.Mean()
    test_accuracy = Accuracy()
    test_accuracy_metric = tf.metrics.Mean()

    def test_step(batch_images, batch_labels, labels_list):
        pred = crnn_model(batch_images, training=False)
        loss_value = loss_object(y_true=batch_labels, y_pred=pred)
        acc = test_accuracy(decoded_text=predict_text(pred), true_label=labels_list)
        loss_metric.update_state(values=loss_value)
        test_accuracy_metric.update_state(values=acc)

    for test_data in test_set:
        batch_images, batch_labels, labels_list = Label().make_label(batch_data=test_data)
        test_step(batch_images, batch_labels, labels_list)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(loss_metric.result(),
                                                           test_accuracy_metric.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy_metric.result() * 100))