import glob
import numpy as np
import os
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split


def main():
    img_size = 50
    dir_names = ["jubeat", "IIDX", "SDVX"]

    if os.path.isfile('./music_game_score_model.h5') is False:  # modelが存在していない場合
        print('create models')

        imgs, labels, names = parse_imgs(dir_names, img_size)
        imgs_train, imgs_test, labels_train, labels_test = train_test_split(imgs, labels, test_size=0.20)

        model = create_model(imgs_train, labels_train)
        model.save('./music_game_score_model.h5')

        print('model save')
        print(model.evaluate(imgs_test, labels_test))

    else:   # modelがすでに存在している場合
        print('load models')

        model = load_model('music_game_score_model.h5')
        test_files = ['jubeat_test', 'IIDX_test', 'SDVX_test']

        imgs_test, labels_test, names = parse_imgs(test_files, img_size)

        predicts = model.predict(imgs_test)
        for n, predict in zip(names, predicts):
            print(n, dir_names[int(np.argmax(predict))])


def parse_imgs(dir_names, img_size):    # 読み込んだ画像とlabelの変換
    imgs = []
    labels = []
    names = []
    for i, name in enumerate(dir_names):
        path = './assets/' + name
        files = glob.glob(path + "/*.JPG")

        for file in files:
            names.append(file)
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((img_size, img_size))
            img_data = np.asarray(image)
            imgs.append(img_data)
            labels.append(i)

    imgs = np.array(imgs)
    labels = np.array(labels)

    imgs = imgs.astype('float32') / 255.0
    labels = np_utils.to_categorical(labels, len(dir_names))  # one-hot形式に変換
    return imgs, labels, names


def create_model(imgs, labels):  # model作成
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=imgs.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    print(model.fit(imgs, labels, epochs=200))

    return model


if __name__ == '__main__':
    main()
