import time
st = time.time()

import os
import numpy as np
import cv2
import math
import json
from imutils import paths
import pandas as pd
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import shutil
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.parquet as pq

using_dictionary = {'uc_userid': 'iu',
                    'data': {
                             'uc_index': [0, 4, 2, 3, 5, 6, 7, 8, 9],
                             'uc_name': ['uc1', 'uc1', 'uc2', 'uc2', 'uc3', 'uc3', 'uc3', 'uc4', 'uc4'],
                             'uc_features_flag': [False, False, False, False, False, False, False, False, False]                        
                             }
                   }
using_json = json.dumps(using_dictionary)


def make_directory(json_input, path):
    information = json.loads(json_input)
    userid = information['uc_userid']
    uc_index = information['data']['uc_index']
    uc_name = information['data']['uc_name']
    uc_features_flag = information['data']['uc_features_flag']
    tmp_dir = os.path.join(path, userid, 'tmp')
    tmp_train_dir = os.path.join(tmp_dir, 'train')
    tmp_test_dir = os.path.join(tmp_dir, 'test')
    features_dir = os.path.join(path, userid, 'cluster_features')
    image_train_dir = os.path.join(path, userid, 'cluster_images')
    if os.path.isdir(tmp_train_dir):
        shutil.rmtree(tmp_train_dir)
    if os.path.isdir(tmp_test_dir):
        shutil.rmtree(tmp_test_dir)
    os.makedirs(tmp_train_dir)
    os.makedirs(tmp_test_dir)

    # 추후에 사용할 것이므로 저장
    flag_false = {}
    flag_false['uc_userid'] = userid
    flag_false_uc_name = []
    flag_false_uc_index = []
    flag_false_tmp_train_path = []
    flag_false_tmp_test_path = []
    
    # flag가 False인 것 추출
    if os.path.isfile(os.path.join(features_dir, 'cluster_class.json')):
        with open(os.path.join(features_dir, 'cluster_class.json'), mode='r', encoding='utf-8') as js:
            cluster_class_dictionary = json.loads(js.read())
    else:
        cluster_class_dictionary = {}

    for inx, flag in enumerate(uc_features_flag):
        if not flag:
            cluster_class_keys = list(cluster_class_dictionary.keys())
            cluster_class_number = len(cluster_class_dictionary)
            if not uc_name[inx] in cluster_class_keys:
                cluster_class_dictionary[uc_name[inx]] = cluster_class_number
#            print(cluster_class_keys)
#            print(cluster_class_number)

            directory_name = str(uc_name[inx]) + '_' + str(uc_index[inx])
            tmp_train_cluster_path = os.path.join(tmp_train_dir, directory_name)
            tmp_test_cluster_path = os.path.join(tmp_test_dir, directory_name)
            features_cluster_dir = os.path.join(features_dir, directory_name)
            # 디렉토리 생성
            os.makedirs(tmp_train_cluster_path)
            os.makedirs(tmp_test_cluster_path)
            if os.path.isdir(features_cluster_dir):
                shutil.rmtree(features_cluster_dir)
            os.makedirs(features_cluster_dir)
            # 추후에 False 것만 쓰기 위해서 추출
            flag_false_uc_name.append(uc_name[inx])
            flag_false_uc_index.append(uc_index[inx])
            flag_false_tmp_train_path.append(tmp_train_cluster_path)
            flag_false_tmp_test_path.append(tmp_test_cluster_path)
    flag_false['uc_index'] = flag_false_uc_index
    flag_false['uc_name'] = flag_false_uc_name
    flag_false['train_path'] = flag_false_tmp_train_path
    flag_false['test_path'] = flag_false_tmp_test_path
    flag_false['path'] = path
    with open(os.path.join(features_dir, 'tmp_cluster_class.json'), mode='w', encoding='utf-8') as js:
        js.write(json.dumps(cluster_class_dictionary))
    return flag_false, cluster_class_dictionary
    
# print(flag_false['uc_index'])
# print(flag_false['uc_name'])
# print(flag_false['train_path'])
# print(flag_false['test_path'])
# print(flag_false['path'])


def load_image(x):
    image_array = []
    for index, img in enumerate(x):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        image[:,:,np.newaxis]
        image = img_to_array(image)
        image_array.append(image)
    image_array = np.array(image_array, dtype="float") / 128. - 1
    return image_array


def make_multiply_number(flag_false, cluster_class_dictionary):
    path = os.path.join(flag_false['path'], flag_false['uc_userid'], 'cluster_images')
#     print(flag_false['path'])
#     print(path)
    uc_index = np.array(flag_false['uc_index'])
    uc_name = np.array(flag_false['uc_name'])
#     print(uc_name)
    tmp_train_path = np.array(flag_false['train_path'])
    tmp_test_path = np.array(flag_false['test_path'])
#     print(tmp_train_path)
#     print(tmp_test_path)

    # 클러스터 이름 확인을 위하여 딕셔너리 keys 이용
    # 클러스터 이름 확인을 위하여 딕셔너리 kerys 이용
    # print(cluster_class_dictionary.keys())
    # print(cluster_class_dictionary)
    multiply_number_dict = {}
    for key in cluster_class_dictionary.keys():
#        print(key)
        # 특정 cluster에 해당하는 index를 찾고자 함
        ind = np.where(uc_name==key)
#        print(ind)
        uc_index_cluster_list = uc_index[ind]
        # Multiply_number 찾기 위해 작업한 것(개수를 합하고 나눠주기)
        cluster_count = 0
        if len(uc_index_cluster_list) != 0:
            for cluster_index in uc_index_cluster_list:
                cluster_path = os.path.join(path, key, str(cluster_index))
#                print(cluster_path)
                path_list = list(paths.list_images(cluster_path))
                path_count = len(path_list)
                cluster_count += path_count
            multiply_number = math.floor(2000 / cluster_count)
            multiply_number_dict[key] = multiply_number
        else:
            multiply_number_dict[key] = 0
    # print(multiply_number_dict['uc1'])
    # print(multiply_number_dict['uc2'])
    return multiply_number_dict


def make_image_load(flag_false, cluster_class_dictionary, multiply_number_dict):
    path = os.path.join(flag_false['path'], flag_false['uc_userid'], 'cluster_images')
    uc_index = np.array(flag_false['uc_index'])
    uc_name = np.array(flag_false['uc_name'])
    tmp_train_path = np.array(flag_false['train_path'])
    tmp_test_path = np.array(flag_false['test_path'])
    # ImageDataGenerator 정의
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=90,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')
    
    # 클러스터 이름 확인을 위하여 딕셔너리 keys 이용
    for key in cluster_class_dictionary.keys():
        # 특정 cluster에 해당하는 index를 찾고자 함
        ind = np.where(uc_name==key)
        uc_index_cluster_list = uc_index[ind]
        multiply_number = multiply_number_dict[key]
        # Multiply_number 찾기위해 작업한 것 (개수를 합하고 나눠주기)
        for cluster_index in uc_index_cluster_list:
            cluster_path = os.path.join(path, key, str(cluster_index))
            path_list = list(paths.list_images(cluster_path))
            path_count = len(path_list)
            # Data split
            x_train, x_test = train_test_split(path_list, test_size=0.2)
            # Data Array Convert
            train_image_array = load_image(x_train)
            test_image_array = load_image(x_test)
            
            # Train Augmentation
            train_rep_num = 0
            test_rep_num = 0
            for batch in train_datagen.flow(train_image_array,
                                           batch_size = len(train_image_array) + 1,
                                           save_to_dir = tmp_train_path[flag_false['uc_index'].index(cluster_index)],
                                           save_prefix='bw', save_format='png'):
                train_rep_num+=1
                if train_rep_num > (multiply_number - 1):
                    break
            
            for batch in train_datagen.flow(test_image_array,
                                           batch_size = len(test_image_array) + 1,
                                           save_to_dir = tmp_test_path[flag_false['uc_index'].index(cluster_index)],
                                           save_prefix='bw', save_format='png'):
                test_rep_num+=1
                if test_rep_num > (multiply_number - 1):
                    break
    return


def load_image3d(x):
    image_array = []
    for index, img in enumerate(x):
        image = cv2.imread(img)
        image[:,:,np.newaxis]
        image = img_to_array(image)
        image_array.append(image)
    image_array = np.array(image_array, dtype="float") / 128. - 1  
    return image_array


def extract_features(flag_false, batch_size):
    features_path = os.path.join(flag_false['path'], flag_false['uc_userid'], 'cluster_features')
    uc_index = np.array(flag_false['uc_index'])
    uc_name = np.array(flag_false['uc_name'])
    tmp_train_path = np.array(flag_false['train_path'])
    tmp_test_path = np.array(flag_false['test_path'])    
    conv_base = MobileNet(weights='imagenet',
                      include_top=False)
    ## 이미지 개수를 저장 
    tmp_train_count = {}
    tmp_test_count = {}
    for inx, cluster_path in enumerate(tmp_train_path) :

        path_list = list(paths.list_images(cluster_path))
        train_image_array = load_image3d(path_list)
        rep_num = math.ceil(len(train_image_array)/batch_size)
        for i in range(rep_num):
            # batch_size*0 :batch_size*1 ,  batch_size*1:batch_size*2 ---->>>> 256:끝
            features = conv_base.predict(train_image_array[i*batch_size:(i+1)*batch_size])
            if i==0:
                features_batch_total = features.copy()
            else:
                features_batch_total = np.vstack([features_batch_total, features])
        tmp_train_count[str(uc_index[inx])] = len(features_batch_total)
        cluster_features_path = os.path.join(features_path, str(uc_name[inx])+'_'+str(uc_index[inx]), 'train.parquet')
        cluster_extract_features= pa.Table.from_arrays([pa.Array.from_pandas(features_batch_total.flatten())], ['array'])
        pq.write_table(cluster_extract_features, cluster_features_path)
        
    for inx, cluster_path in enumerate(tmp_test_path) :

        path_list = list(paths.list_images(cluster_path))
        test_image_array = load_image3d(path_list)
        rep_num = math.ceil(len(test_image_array)/batch_size)
        for i in range(rep_num):
            train_features = conv_base.predict(test_image_array[i*batch_size:(i+1)*batch_size])
            if i==0:
                features_batch_total = features.copy()
            else:
                features_batch_total = np.vstack([features_batch_total, train_features])
        tmp_test_count[str(uc_index[inx])] = len(features_batch_total)
        cluster_features_path = os.path.join(features_path, str(uc_name[inx])+'_'+str(uc_index[inx]), 'test.parquet')
        cluster_extract_features= pa.Table.from_arrays([pa.Array.from_pandas(features_batch_total.flatten())], ['array'])
        pq.write_table(cluster_extract_features, cluster_features_path)
    
    class_json_path = os.path.join(features_path,'tmp_cluster_class.json')
    target_class_json_path = os.path.join(features_path,'cluster_class.json')
    shutil.move(class_json_path, target_class_json_path)
    return tmp_train_count, tmp_test_count


def show_features(flag_false, total_json):
    features_path = os.path.join(flag_false['path'], flag_false['uc_userid'], 'cluster_features')
    
    with open(os.path.join(features_path, 'cluster_class.json'), mode='r', encoding='utf-8') as js:
        class_label_dict = json.loads(js.read())
    total_json = json.loads(total_json)['data']
    true_uc_index = total_json['uc_index']
    true_uc_name = total_json['uc_name']
    
    ## True Data Load
    if len(true_uc_name) != 0:
        for inx, cluster_name in enumerate(true_uc_name):
            label_encoding_number = class_label_dict[cluster_name]
            path = os.path.join(features_path, str(cluster_name) + '_' + str(true_uc_index[inx]))
            train_path = os.path.join(path, 'train.parquet')
            test_path = os.path.join(path, 'test.parquet')
            if inx == 0:
                train = pq.read_table(train_path).to_pandas()['array'].values
                train_num = int(len(train)/(3*3*1024))
                train = train.reshape(train_num, 3, 3, 1024)
                train_label_array = np.array([label_encoding_number] * train_num)
                print(cluster_name, train_num, len(train_label_array))
            else:
                train_part = pq.read_table(train_path).to_pandas()['array'].values
                train_num = int(len(train_part)/(3*3*1024))
                train = np.concatenate([train, train_part.reshape(train_num, 3, 3, 1024)])
                train_label_array = np.concatenate([train_label_array, np.array([label_encoding_number]*train_num)])
        
            if inx == 0:
                test = pq.read_table(test_path).to_pandas()['array'].values
                test_num = int(len(test)/(3*3*1024))
                test = test.reshape(test_num, 3, 3, 1024)
                test_label_array = np.array([label_encoding_number] * test_num)
            else:
                test_part = pq.read_table(test_path).to_pandas()['array'].values
                test_num = int(len(test_part)/(3*3*1024))
                test = np.concatenate([test, test_part.reshape(test_num, 3, 3, 1024)])
                test_label_array = np.concatenate([test_label_array, np.array([label_encoding_number] * test_num)])

    return train, train_label_array, test, test_label_array


def model_training(train_feature_array, train_label_array, test_feature_array, test_label_array,
                   cluster_class_dictionary, flag_false, epoch=10):
    cluster_name_unique = len(cluster_class_dictionary)
    features_path = os.path.join(flag_false['path'], flag_false['uc_userid'], 'cluster_features')
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(3, 3, 1024)))
    model.add(layers.Dense(256, activation='relu', input_dim=(3*3*512)))
    model.add(layers.Dropout(0.5))
    if cluster_name_unique >= 3:
        model.add(layers.Dense(cluster_name_unique, activation='softmax'))
        model.compile(optimizers.RMSprop(lr=2e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])
    elif cluster_name_unique == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizers.RMSprop(lr=2e-5),
                      loss='binary_crossentropy',
                      metrics=['acc'])
    else:
        print("클러스터의 개수가 최소 2개 이상 필요합니다.")
    
    model.fit(train_feature_array, train_label_array,
              epochs = epoch,
              batch_size = 128,
              shuffle = True,
              validation_data = (test_feature_array, test_label_array))
    model.save_weights(os.path.join(features_path, "model.h5"))
    features_path = os.path.join(flag_false['path'], flag_false['uc_userid'], 'tmp')
    shutil.rmtree(features_path)
    return {"success": True}


# Execute
path = '/root/example/Py_example/my_cluster_classification/'
flag_false, cluster_class_dictionary= make_directory(using_json, path)
multiply_number = make_multiply_number(flag_false, cluster_class_dictionary)
make_image_load(flag_false, cluster_class_dictionary, multiply_number)
tmp_train_count, tmp_test_count = extract_features(flag_false, 128)
train_feature_array, train_label_array, test_feature_array,test_label_array =show_features(flag_false, using_json)
model_training(train_feature_array, train_label_array, test_feature_array, test_label_array,
               cluster_class_dictionary, flag_false, 10)
en = time.time()
print("Full TIME:", en - st)