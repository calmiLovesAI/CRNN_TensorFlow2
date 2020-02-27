import tensorflow as tf
from core.read_dataset import Dataset
from core.make_label import Label
from core.crnn import CRNN
from core.loss import CTCLoss
from core.metric import Accuracy
from core.predict import predict_text
from configuration import BATCH_SIZE, EPOCHS, save_model_dir, save_frequency


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # dataset
    dataset = Dataset()
    train_set, train_size = dataset.train_dataset()
    valid_set, valid_size = dataset.valid_dataset()

    # model
    crnn_model = CRNN()

    # loss
    loss = CTCLoss()

    # optimizer
    optimizer = tf.optimizers.Adadelta()

    # metrics
    train_loss_metric = tf.metrics.Mean()
    valid_loss_metric = tf.metrics.Mean()
    accuracy = Accuracy()
    accuracy_metric = tf.metrics.Mean()

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = crnn_model(batch_images, training=True)
            loss_value = loss(y_true=batch_labels, y_pred=pred)
        gradients = tape.gradient(target=loss_value, sources=crnn_model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, crnn_model.trainable_variables))
        train_loss_metric.update_state(values=loss_value)

    def valid_step(batch_images, batch_labels, labels_list):
        pred = crnn_model(batch_images, training=False)
        loss_value = loss(y_true=batch_labels, y_pred=pred)
        acc = accuracy(decoded_text=predict_text(pred), true_label=labels_list)
        valid_loss_metric.update_state(values=loss_value)
        accuracy_metric.update_state(values=acc)


    for epoch in range(EPOCHS):
        for step, train_data in enumerate(train_set):
            batch_images, batch_labels, _ = Label().make_label(batch_data=train_data)
            train_step(batch_images, batch_labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}".format(epoch,
                                                                   EPOCHS,
                                                                   step,
                                                                   tf.math.ceil(train_size / BATCH_SIZE),
                                                                   train_loss_metric.result()))

        for valid_data in valid_set:
            batch_images, batch_labels, labels_list = Label().make_label(batch_data=valid_data)
            valid_step(batch_images, batch_labels, labels_list)
        print("Epoch: {}/{}, valid_loss: {:.5f}, valid_accuracy: {:.5f}".format(epoch,
                                                                                EPOCHS,
                                                                                valid_loss_metric.result(),
                                                                                accuracy_metric.result()))

        train_loss_metric.reset_states()
        valid_loss_metric.reset_states()
        accuracy_metric.reset_states()

        if epoch % save_frequency == 0:
            crnn_model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format="tf")

    crnn_model.save_weights(filepath=save_model_dir+"saved_model", save_format="tf")
