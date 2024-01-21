# the code here is extracted from 00_annotations
import pandas as pd
from PIL import Image
import os

script_dir = os.path.dirname(__file__)
classes = ["N", "D", "G", "C", "A", "H", "M", "O"]
classes_name = ["normal", "diabetes", "glaucoma", "cataract", "AMD", "hypertensi\
    on", "myopia", "other diseases"]

def load_annotation():
    df = pd.read_csv(os.path.join(script_dir, "../annotations.csv"), index_col=0)
    return df

def load_left_eye_image(df, i, class_=None ):
    if class_==None:
        return Image.open(os.path.join(script_dir, "../images/{}").format(df.Left_Fundus.values[i]))
    image = Image.open(os.path.join(script_dir, "../images/{}").format(df[df[class_] == 1].Left_Fundus.values[i]))
    return image

def load_right_eye_image(df,i, class_=None):
    if class_==None:
        return Image.open(os.path.join(script_dir, "../images/{}").format(df.Right_Fundus.values[i]))
    image = Image.open(os.path.join(script_dir, "../images/{}").format(df[df[class_] == 1].Right_Fundus.values[i]))
    return image

if __name__ == "__main__":
    df = load_annotation()
    print("Head of the annotations")
    print(df.head())
    image = load_left_eye_image(df, class_="N", i = 0)
    image.thumbnail((200,200))
    image.save("a_normal_left_eye.jpg", "JPEG")
