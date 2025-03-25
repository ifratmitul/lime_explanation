def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_test