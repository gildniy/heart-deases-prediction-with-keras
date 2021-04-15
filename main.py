# # %%
# #  EXERCISE 1
# import numpy as np
# from tensorflow import keras
#
# model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mse')
#
# xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
#
# model.fit(xs, ys, epochs=500)
# print(model.predict([10.0]))
#
# # %%

# # %%
# #  EXERCISE 2
# import tensorflow as tf
#
# print(tf.__version__)
#
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (
#     test_images, test_labels) = mnist.load_data()
#
# import matplotlib.pyplot as plt
#
# plt.imshow(training_images[0])
#
# # print(training_labels[0])
# # print(test_images[0])
#
# training_images = training_images / 255.
# test_images = test_images / 255.
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.elu),
#     # tf.keras.layers.Dense(1024, activation=tf.nn.elu),
#     # tf.keras.layers.Dense(256, activation=tf.nn.elu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax),
# ])
#
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# model.fit(training_images, training_labels, epochs=30)
# loss, accuracy = model.evaluate(test_images, test_labels)
#
# print("\nLoss: ", loss)
# print("Accuracy: ", accuracy)

# %%
#
# # %%
# #  EXERCISE 3
# import tensorflow as tf
#
# print(tf.__version__)
#
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (
#     test_images, test_labels) = mnist.load_data()
#
# training_images = training_images.reshape(60000, 28, 28, 1)
# training_images = training_images / 255.
#
# test_images = test_images.reshape(10000, 28, 28, 1)
# test_images = test_images / 255.
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,
#                                                                        28, 1)),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax'),
# ])
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()
#
# model.fit(training_images, training_labels, epochs=5)
# loss, accuracy = model.evaluate(test_images, test_labels)
#
#
# print("\nLoss: ", loss)
# print("Accuracy: ", accuracy)
# %%

# %%
#  EXERCISE 4

# Commencez par importer quelques bibliothèques Python et l'image « ascent » :

import cv2
import numpy as np
from scipy import misc

i = misc.ascent()

# Ensuite, utilisez la bibliothèque Pyplot matplotlib pour dessiner l'image afin que vous sachiez
# à quoi elle ressemble :

import matplotlib.pyplot as plt

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()
#  résultat:
#
# #  [image]
#
# # Vous pouvez voir que c'est l'image d'une cage d'escalier. Il existe de nombreuses
# # fonctionnalités que vous pouvez essayer et isoler. Par exemple, il y a de fortes lignes verticales.
# # L'image est stockée sous forme de tableau NumPy, nous pouvons donc créer l'image
# # transformée en copiant simplement ce tableau. Les variables size_x et size_y contiendront les
# # dimensions de l'image afin que vous puissiez la parcourir plus tard.

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# Etape 2 :
# Créez une matrice de convolution sous la forme d'un tableau 3x3.

# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] -> “Select one of the filters”


# Maintenant, calculez les pixels de sortie. Itérez sur l'image en laissant une marge de 1 pixel, et
# multipliez chacun des voisins du pixel actuel par la valeur définie dans le filtre. Cela signifie
# que le voisin du pixel actuel au-dessus et à gauche de celui-ci sera multiplié par l'élément en
# haut à gauche du filtre. Ensuite, multipliez le résultat par le poids et assurez-vous que le
# résultat est compris entre 0 et 255. Enfin, chargez la nouvelle valeur dans l'image transformée:


weight = 1

for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
        convolution = 0.0
    convolution = convolution + (i[x - 1, y - 1] * filter[0][0])
    convolution = convolution + (i[x, y - 1] * filter[0][1])
    convolution = convolution + (i[x + 1, y - 1] * filter[0][2])
    convolution = convolution + (i[x - 1, y] * filter[1][0])
    convolution = convolution + (i[x, y] * filter[1][1])
    convolution = convolution + (i[x + 1, y] * filter[1][2])
    convolution = convolution + (i[x - 1, y + 1] * filter[2][0])
    convolution = convolution + (i[x, y + 1] * filter[2][1])
    convolution = convolution + (i[x + 1, y + 1] * filter[2][2])
    convolution = convolution * weight
    if convolution < 0:
        convolution = 0
    if convolution > 255:
        convolution = 255
    i_transformed[x, y] = convolution

# Etape 3:
# Maintenant, tracez l'image pour voir l'effet de passer le filtre dessus.
# <Pooling> Note the size of the axes -- they are 512 by 512
### Tracez l'image. Notez la taille des axes - ils sont 512 par 512

plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
# plt.axis('off')
plt.show()

# Tenez compte des valeurs de filtre suivantes et de leur impact sur l'image. L'utilisation de [-1,0,1,-2,0,2,
# -1,0,1] vous donne un ensemble très fort de lignes verticales ;

# [image]


# Etape 4 : comprendre le pooling
# Maintenant que vous avez identifié les caractéristiques essentielles de l'image, que faitesvous?
# Comment utilisez-vous la carte des caractéristiques résultante pour
# classer les images?
# Semblable aux convolutions, la mise en commun (pooling) facilite grandement la détection
# des fonctionnalités. La mise en commun des couches réduit la quantité globale d'informations
# dans une image tout en conservant les caractéristiques détectées comme présentes. vous
# utiliserez une fonction appelé regroupement maximal. Parcourez l'image et, à chaque point,
# considérez le pixel et ses voisins immédiats à droite, en dessous et juste en dessous. Prenez le
# plus grand de ceux-ci (d'où le pooling maximal) et chargez-le dans la nouvelle image. Ainsi, la
# nouvelle image aura un quart de la taille de l'ancienne.

# [image]

# Le code suivant affichera un regroupement (2, 2). Exécutez-le pour voir la sortie. Vous verrez
# que si l'image fait un quart de la taille de l'original, elle a conservé toutes les fonctionnalités.

new_x = int(size_x / 2)
new_y = int(size_y / 2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x + 1, y])
    pixels.append(i_transformed[x, y + 1])
    pixels.append(i_transformed[x + 1, y + 1])
    pixels.sort(reverse=True)
    newImage[int(x / 2), int(y / 2)] = pixels[0]
# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
# plt.axis('off')
plt.show()


# [image]

# L'image mesure désormais 256x256, soit un quart de sa taille d'origine, et les caractéristiques
# détectées ont été améliorées malgré moins de données dans l'image.

# %%

# # %%
# #  EXERCISE 5
# # How to Predict Sentiment From Movie Reviews Using Deep Learning (Text Classification)
# # https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
# # https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true
# import tensorflow as tf
# from keras.datasets import imdb
# import numpy as np
#
# print(tf.__version__)
#
# (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz")
# X = np.concatenate((X_train, X_test), axis=0)
# y = np.concatenate((y_train, y_test), axis=0)
#
# # summarize size
# print("Training data: ")
# print(X.shape)
# print(y.shape)
# # %%
#

# https://www.kaggle.com/luigisaetta/tab-playground-nb2
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers.experimental.preprocessing import \
#     Normalization, CategoryEncoding, IntegerLookup
#
# dataset_url = 'https://storage.googleapis.com/kaggle-data-sets/1226038/2047221/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210414%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210414T143749Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=6cefbf3a9411483ae20d76e70c31f465c92380544351145764bdfb678d88c27622dee6c47cb9d2504859c08df24648c576fb1b8d8af192263f8c2b6485157cca2751d166fdaf358243774c4eaa1677c561af42961344b5c011e8bbfdd8fc1237f1101658d29c57b10cd1e9a5bb816461e05ff2190b520a5ae115147203e2e9c2a042787e26e6c314431e9caad65c7203419c3c52097f1ab00416a928aecba4ee78d2b559d1ea29c0ba68736c7ed2c630ef4574092aec1825f36f228f7aacdc44122ec8bff18a6051d9d74a9445924ceec8d0edf1de4741b465766876f4e27a491a20aded988467b4cefa75661d79b0d322e49d6985887307459dae559af523e6'
# tf.keras.utils.get_file('archive.zip', dataset_url, extract=True,
#                         cache_dir='.')
# dataframe = pd.read_csv('datasets/heart.csv')
#
# print(dataframe.shape)  # (303, 14)
# print(dataframe.head())
#
# train_df, val_df = train_test_split(dataframe, test_size=0.2)
# train_df, pred_df = train_test_split(train_df, test_size=0.01)
#
#
# def df_to_dataset(df, predictor, shuffle=True, batch_size=32):
#     df = df.copy()
#     labels = df.pop(predictor)
#     ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(df))
#     ds = ds.batch(batch_size)
#     return ds
#
#
# train_ds = df_to_dataset(train_df, 'output', batch_size=25)
# val_ds = df_to_dataset(val_df, 'output', shuffle=False, batch_size=25)
#
#
# def get_normalization_layer(name, dataset):
#     normalizer = Normalization()
#     feature_ds = dataset.map(lambda x, y: x[name])
#     normalizer.adapt(feature_ds)
#
#     return normalizer
#
#
# def get_category_encoding_layer(name, dataset, max_tokens=None):
#     index = IntegerLookup(max_tokens=max_tokens)
#     feature_ds = dataset.map(lambda x, y: x[name])
#     index.adapt(feature_ds)
#     encoder = CategoryEncoding(num_tokens=index.vocabulary_size())
#     feature_ds = feature_ds.map(index)
#     encoder.adapt(feature_ds)
#
#     return lambda feature: encoder(index(feature))
#
#
# all_inputs = []
# encoded_features = []
#
# numeric_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'slp']
#
# for header in numeric_cols:
#     numeric_col = tf.keras.Input(shape=(1,), name=header)
#     all_inputs.append(numeric_col)
#
#     normalization_layer = get_normalization_layer(header, train_ds)
#     encoded_numeric_col = normalization_layer(numeric_col)
#     encoded_features.append(encoded_numeric_col)
#
# categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'caa', 'thall']
#
# for header in categorical_cols:
#     categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
#     all_inputs.append(categorical_col)
#
#     encoding_layer = get_category_encoding_layer(
#         header,
#         train_ds,
#         max_tokens=5
#     )
#     encoded_categorical_col = encoding_layer(categorical_col)
#     encoded_features.append(encoded_categorical_col)
#
#
# def build_model(n_units):
#     all_features = tf.keras.layers.concatenate(encoded_features)
#     x = tf.keras.layers.Dense(n_units, activation="relu")(all_features)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#     model = tf.keras.Model(all_inputs, output)
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.BinaryCrossentropy(),
#                   metrics=["accuracy"])
#     return model
#
#
# model = build_model(32)
#
# tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
#
# model.summary()
#
# model.fit(train_ds, validation_data=val_ds, epochs=50)
#
# loss, accuracy = model.evaluate(val_ds)
#
# print("\nLoss: ", loss)
# print("Accuracy: ", accuracy)
#
# sample = list(pred_df.to_dict('index').values())[0]
#
# input_dict = {name: tf.convert_to_tensor([value]) for name, value in
#               sample.items() if name != 'output'}
# predictions = model.predict(input_dict)
#
# print(f"\nThis particular patient had a {100 * predictions[0][0]:.1f}% "
#       f"probability of having a heart disease, as evaluated by our model.")
