# import keras
# import tensorflow as tf
from keras import layers, Input, applications
# from keras import applications
# from keras import models
from keras.models import Model
import os
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16 as KerasVGG16
# from keras import optimizers
from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD


# from keras.applications.resnet50 import ResNet50 as KerasResNet50


# pretrained models used = 1) VGG16 2) INCEPTION
# @TODO: train the models from scratch + save the full results for each run. // for today // validation 12.455% - testing 12.455% - training  75.457%
# @TODO: train the models from scratch with different split of the training - validation - testing datasets + save the full results for each run. // for tmrw
# @TODO: BUILD PDF FILE and submit file


# dataset was taken from kaggle.com made by oxford
# https://www.kaggle.com/nunenuh/pytorch-challange-flower-dataset?select=sample_submission.csv


def DataSets():
    # getting the path of the dataset/test set/train set/validation set
    global testSet, trainSet, validSet
    DataSet = os.path.join(Main_Dir, "dataset")
    testSet = os.path.join(DataSet, "test")
    trainSet = os.path.join(DataSet, "train")
    validSet = os.path.join(DataSet, "valid")


def trainGen(imageSize):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        trainSet,
        target_size=imageSize,
        batch_size=16,
        class_mode='categorical')
    return train_generator


def validGen(imageSize):
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = valid_datagen.flow_from_directory(
        validSet,
        target_size=imageSize,
        batch_size=16,
        class_mode='categorical')
    return validation_generator


def testGen(imageSize):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        testSet,
        target_size=imageSize,
        batch_size=16,
        class_mode='categorical')
    return test_generator


def CreateVGG16Model():
    base_model = KerasVGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224) + (3,)))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='elu', name='fc1')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='elu', name='ch1')(x)
    x = Dropout(0.6)(x)
    predictions = Dense(102, activation='softmax', name='predictions')(x)
    model = Model(base_model.input, predictions)
    return model


def DisplayResults(modelstr):
    plt.plot(epochs, accuracy, 'bo', label='Training acc')
    plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('ACC-' + modelstr + '.png')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('LOSS-' + modelstr + '.png')
    plt.show()
    # Evaludate the model using the test set.
    score = model.evaluate(test_generator, steps=30)
    print("The test loss is ", score[0])
    print("The test accuracy is ", score[1])


def InceptionModel():
    base_model = KerasInceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299) + (3,)))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='elu', name='fc1')(x)
    predictions = Dense(102, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    return model


# def RESNETModel():
#     base_model = applications.resnet50.ResNet50(weights=None,include_top=False, input_shape=(224, 224, 3))
#     for layer in base_model.layers:
#         layer.trainable = False
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.7)(x)
#     predictions = Dense(102, activation='softmax')(x)
#     model = Model(base_model.input, predictions)
#     return model


if __name__ == '__main__':
    int lmao
    Main_Dir = os.getcwd()
    DataSets()

    # choose model
    in1 = input('choose which model you want to train:\n'
                '1)VGG16\n'
                '2)inception\n')

    if in1 == 'VGG16' or in1 == '1':

        train_generator = trainGen((224, 224))
        validation_generator = validGen((224, 224))
        test_generator = testGen((224, 224))

        # VGG16 MODEL
        model = CreateVGG16Model()

        if os.path.isfile('VGG16Model.h5'):
            model.load_weights('VGG16Model.h5')

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])
        model.summary()

        history = model.fit(
            train_generator,
            steps_per_epoch=50,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=30)

        model.save('VGG16Model.h5')

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(accuracy))
        DisplayResults('VGG16')

    elif in1 == 'inception' or in1 == '2':

        train_generator = trainGen((299, 299))
        validation_generator = validGen((299, 299))
        test_generator = testGen((299, 299))

        # INCEPTION MODEL
        print('CREATING INCEPTION MODEL \nRESULTS WILL BE BEGINNING TO DISPLAY SHORTLY\n')
        model = InceptionModel()

        for layer in model.layers[:80]:
            layer.trainable = False
        for layer in model.layers[80:]:
            layer.trainable = True

        if os.path.isfile('inceptionModel.h5'):
            model.load_weights('inceptionModel.h5')

        model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])

        model.summary()

        history = model.fit(
            train_generator,
            steps_per_epoch=50,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=30)

        model.save('inceptionModel.h5')

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(accuracy))
        DisplayResults('INCEPTION')


    # elif in1 == 'resnet' or in1 == '3':
    #     train_generator = trainGen((224, 224))
    #     validation_generator = validGen((224, 224))
    #     test_generator = testGen((224, 224))
    #
    #     print('CREATING RESNET MODEL \nRESULTS WILL BE BEGINNING TO DISPLAY SHORTLY\n')
    #
    #     model = RESNETModel()
    #     for layer in model.layers[:80]:
    #         layer.trainable = False
    #     for layer in model.layers[80:]:
    #         layer.trainable = True
    #
    #     if os.path.isfile('flower_classification3.h5'):
    #         model.load_weights('flower_classification3.h5')
    #
    #     model.compile(
    #         loss='categorical_crossentropy',
    #         optimizer=Adam(lr=1e-5),
    #         metrics=['accuracy'])
    #
    #     model.summary()
    #
    #     history = model.fit(
    #         train_generator,
    #         steps_per_epoch=50,
    #         epochs=30,
    #         validation_data=validation_generator,
    #         validation_steps=30)
    #
    #     model.save('flower_classification3.h5')
    #
    #     accuracy = history.history['accuracy']
    #     val_accuracy = history.history['val_accuracy']
    #     loss = history.history['loss']
    #     val_loss = history.history['val_loss']
    #     epochs = range(len(accuracy))
    #     DisplayResults('RESNET')

    else:
        print('wrong input , try again next time')
