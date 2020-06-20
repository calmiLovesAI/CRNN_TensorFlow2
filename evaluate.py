import tensorflow as tf
from core.read_dataset import Dataset
from core.make_label import Label
from core.crnn import CRNN
from core.loss import CTCLoss
from core.metric import Accuracy
from core.predict import predict_text
from configuration import Config


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # dataset
    dataset = Dataset()
    num_classes = dataset.num_classes
    blank_index = dataset.blank_index
    test_set, test_size = dataset.test_dataset()

    # model
    crnn_model = CRNN(num_classes)
    crnn_model.load_weights(filepath=Config.save_model_dir+"saved_model")

    # loss and metrics
    loss_object = CTCLoss()
    loss_metric = tf.metrics.Mean()
    test_accuracy = Accuracy(blank_index)
    test_accuracy_metric = tf.metrics.Mean()

    def test_step(batch_images, batch_labels):
        pred = crnn_model(batch_images, training=False)
        loss_value = loss_object(y_true=batch_labels, y_pred=pred)
        acc = test_accuracy(decoded_text=predict_text(pred, blank_index=blank_index), true_label=batch_labels)
        loss_metric.update_state(values=loss_value)
        test_accuracy_metric.update_state(values=acc)

    for test_data in test_set:
        batch_images, batch_labels = Label().make_label(batch_data=test_data)
        test_step(batch_images, batch_labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(loss_metric.result(),
                                                           test_accuracy_metric.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy_metric.result() * 100))