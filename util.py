import numpy as np


def classify(image, model, class_names):

    image = image.resize((224, 224))
    image_array = np.asarray(image)

    normalized_image_array = image_array.astype(np.float32)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    return class_name
