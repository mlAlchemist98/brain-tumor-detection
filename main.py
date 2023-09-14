import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify

# set title
st.title("Распознание видов опухоли мозга по снимку МРТ")

# set header
st.header("Загрузите пожалуйста сканированный снимок МРТ. Результат анализа отобразится снизу")

# file uploading
file = st.file_uploader("", type=["jpeg", "png", "jpg"])

# load classifier
model = load_model("./model/brain_tumor_detection_model.h5")

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a.strip().split(" ")[1] for a in f.readlines()]
    f.close()

# display image
if file:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name = classify(image=image, model=model, class_names=class_names)

    if class_name == "Glioma":
        st.write(f"### Согласно модели классификации, мозг на снимке поражен глиомой")
    elif class_name == "Meningioma":
        st.write(f"### Согласно модели классификации, мозг на снимке поражен менингиомой")
    elif class_name == "Healthy":
        st.write(f"### Согласно модели классификации, мозг на снимке не имеет никаких злокачественных опухолей")
    else:
        st.write(f"### Согласно модели классификации, мозг на снимке поражен опухолью гипофиза")

st.write("Модель классификации использует данные со следующих источников:")
st.write("https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")
st.write("https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri")
st.write("https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no")
