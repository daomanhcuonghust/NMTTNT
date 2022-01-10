import os
import shutil


# 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
fer_path = 'datasets/fer2013/images_fer2013'

fer_path_train = os.path.join(fer_path, "Training")
fer_path_valid = os.path.join(fer_path, "PublicTest")
fer_path_test = os.path.join(fer_path, "PrivateTest")
dict_label = {'Surprise': 1, 'Fear': 2, 'Disgust': 3, 'Happy': 4, 'Sad': 5, 'Angry': 6, 'Neutral': 7}
dict_train = {}
dict_valid = {}
dict_test = {}

image_path = 'datasets/fer2013/images_fer2013/Image'
os.mkdir(image_path)

for folder in os.listdir(fer_path_train):
    iter = 0
    for image in os.listdir(os.path.join(fer_path_train, folder)):
        if iter == 0: 
            iter += 1
            continue
        # print(image)
        # print(type(image))
        dict_train[image] = dict_label[folder]
        shutil.move(os.path.join(fer_path_train, folder, image), image_path)


for folder in os.listdir(fer_path_valid):
    iter = 0
    for image in os.listdir(os.path.join(fer_path_valid, folder)):
        if iter == 0: 
            iter += 1
            continue
        # print(image)
        # print(type(image))
        dict_valid[image] = dict_label[folder]
        shutil.move(os.path.join(fer_path_valid, folder, image), image_path)


for folder in os.listdir(fer_path_test):
    iter = 0
    for image in os.listdir(os.path.join(fer_path_test, folder)):
        if iter == 0: 
            iter += 1
            continue
        # print(image)
        # print(type(image))
        dict_test[image] = dict_label[folder]


with open('datasets/fer2013/images_fer2013/list_patition_label.txt', 'w') as f:
    for key in sorted(dict_train.keys()):
        line = 'train_' + str(key) + ' ' + str(dict_train[key]) + '\n'
        f.write(line)

    for key in sorted(dict_valid.keys()):
        line = 'test_' + str(key) + ' ' + str(dict_valid[key]) + '\n'
        f.write(line)