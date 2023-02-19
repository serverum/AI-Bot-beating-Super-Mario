import cv2
import numpy as np
import base64
# from MarioGame import MarioGame
import tensorflow as tf
from tensorflow.keras.models import Sequential
import joblib
import os

example = cv2.imread("example.png")
mario_image = cv2.imread("mario_image.png")
filepath = os.path.dirname(os.path.realpath(__file__))
# cv2.imshow('', mario_image)
# cv2.waitKey(0)


# функция для преобразования координаты точки, и ширины и высоты, в 2 две координаты и обратно

def intTuple(tup):
    return tuple([int(x) for x in tup])


def boxToRectangle(bound_box):
    bound_box = intTuple(bound_box)
    top_left = (bound_box[0], bound_box[1])
    bottom_right = (bound_box[0] + bound_box[2], bound_box[1] + bound_box[3])
    return top_left, bottom_right


def rectangleToBox(top_left, bottom_right):
    result = (
        top_left[0],
        top_left[1],
        bottom_right[0] - top_left[0],
        bottom_right[1] - top_left[1]
    )
    return intTuple(result)


def save_screenshots(frame, n):  # frame - картинка, n - номер кадра
    if n > 30:
        return
    print(f"{n}", end=" ")
    with open(f"/content/screenshots/{n}.png", "wb") as screen_file:
        frame_bytes = base64.b64decode(frame)
        screen_file.write(frame_bytes)


# input_image - картинка на которой ищем,
# template_image - картинка та которую ищем
input_image = example
template_image = mario_image
BGR_YELLOW = (0, 255, 255)


def matchTemplate(input_image, template_image, method=cv2.TM_CCOEFF):
    input_image_copy = input_image.copy()
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    template_image = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)

    matching_result = cv2.matchTemplate(input_image, template_image, method)

    _, _, _, top_left = cv2.minMaxLoc(
        matching_result)  # прочитать в документации за функцию и 4 значения что возвращает minMaxLoc()

    h, w = template_image.shape

    button_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(input_image_copy, top_left, button_right, BGR_YELLOW, 2)

    box = rectangleToBox(top_left, button_right)

    return input_image_copy, box


img, box = matchTemplate(example, template_image)

# print(box)
# cv2.imshow('', img)
# cv2.waitKey(0)

# mario = MarioGame(autoplay=True, callback=save_screenshots, update_interval=50)  # указать настройки
# mario.runGame()

# Основая работа с нейронными сетями через керас

image_size = (352, 40)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "classification/dataset",
    validation_split=0.10,
    image_size=image_size,
    subset="training",
    seed=42690

)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "classification/dataset",
    validation_split=0.15,
    image_size=image_size,
    subset="validation",
    seed=42690

)

class_names = train_ds.class_names

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
images, labels = next(iter(normalized_ds))

print(class_names)

class_counts = len(class_names)

cnn_model = Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255, input_shape=(352, 40, 3)),

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(class_counts)
])

cnn_model.summary()

cnn_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

epoch_count = 2

# cnn_model.fit(train_ds, validation_data=validation_ds, epochs=epoch_count)
#
# cnn_model.save(filepath)

cnn_model = tf.keras.models.load_model(filepath)

# cnn_model.save_weights(filepath)

def classifyImage(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = cnn_model.predict(img_array)

    score = tf.nn.softmax(predictions)
    print(class_names[np.argmax(score)], 100 * np.max(score))
    # print(predictions)

# classifyImage('classification/simple/TURTLE/screen115_cut9.png') черепаху определяет как облако, ошибки 0.5

# classifyImage('classification/simple/MARIO/screen190_cut8.png') - хорошо ищет марио

classifyImage('classification/dataset/FLOWER/screen184_cut6.png')

