import os
import sys
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
from train_model import create_model


def load_model(model_name):
    model = create_model()
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    weights_path = os.path.join(project_dir, "models", model_name + ".hdf5")
    model.load_weights(weights_path)
    return model


def split_image(image):
    width = image.size[0]
    height = image.size[1]

    num_horiz = width - 40
    num_vert = height - 20
    test_x = np.zeros((num_horiz * num_vert, 20, 40, 3))
    i = 0

    for y_offset in range(0, height - 20):
        for x_offset in range(0, width - 40):
            x_bound = x_offset + 40
            y_bound = y_offset + 20

            image_tile = image.crop([x_offset, y_offset, x_bound, y_bound])
            img_array = img_to_array(image_tile) / 255
            test_x[i] = img_array
            i += 1
    return test_x, num_horiz, num_vert


def recombine_image(pieces, num_horiz, num_vert):
    full_img = Image.new("RGB", [num_horiz + 40, num_vert + 20], color=None)
    for i, piece in enumerate(pieces):
        horiz_offset = i % num_horiz
        vert_offset = i // num_horiz
        img_tile = array_to_img(piece)
        temp_img = full_img.copy()
        temp_img.paste(img_tile, [horiz_offset, vert_offset])
        full_img = Image.blend(full_img, temp_img, 0.5)

    full_img = full_img.crop(full_img.getbbox())
    return full_img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need to supply a model name!")
        sys.exit()
    model_name = sys.argv[1]
    model = load_model(model_name)

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    test_data_path = os.path.join(project_dir, "data", "raw", "test")
    test_filenames = os.listdir(test_data_path)
    for test_image_name in test_filenames:
        test_image = load_img(os.path.join(test_data_path, test_image_name))
        test_x, num_horiz, num_vert = split_image(test_image)

        test_y = model.predict(test_x)

        full_image = recombine_image(test_y, num_horiz, num_vert)
        full_image.save(os.path.join(project_dir, "data", "predictions", test_image_name))
