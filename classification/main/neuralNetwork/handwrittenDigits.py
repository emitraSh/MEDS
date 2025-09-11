import neuralNetworkFromScratch
import mnist_loader



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = neuralNetworkFromScratch.Network([784, 30, 10])

net.SGD(df_train =training_data,
        epochs =30,
        mini_batch_size =10,
        learning_rate =3.0,
        df_test =test_data)