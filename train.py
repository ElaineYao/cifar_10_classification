from utils import *
from network import *

def train():
    (train_X, train_Y, test_X, test_Y) = preprocess()
    model = network()
    # There is no validation data. We use test data to get the accuracy
    model.fit(train_X, train_Y,
              validation_data=(test_X, test_Y),
              epochs=5, batch_size=32)
    # Calculate its accuracy on testing data
    _,acc = model.evaluate(test_X, test_Y)
    print('The accuracy on the testing data is {}.'.format(acc*100))
    # Save the model
    model.save('model1_cifar_5epoch.h5')


if __name__ == '__main__':
    train()