# !pip install easyocr
import easyocr
#download the model
reader = easyocr.Reader(['en'], gpu = True)
# show an image
import PIL
from PIL import ImageDraw
from PIL import Image
import cv2
import PIL
from PIL import ImageDraw
from PIL import Image
import cv2
import pandas as pd
from pandas import DataFrame
import pandas as pd
import json
import glob
# import xlrd 

filenames = glob.glob("*.jpg")
filenames.sort()
# filenames = filenames[:50]
# Draw bounding boxes
def draw_boxes(image, bounds, color='green', width=1):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color , width=width)
        # if bound[1] == "ToTAL" or bound[1] =="TOTAL" or bound[1]=="TOTAL" or bound[1] =="Total Payable;" or bound[1] =="Total Payable:" or bound[1] =="Total Payable:" or bound[1]=='Total' or bound[1]=='TOTAL' or bound[1]=="Totz' Ingi, 0f GST" or bound[1]=="Total Sales (Inclusive of GST)" or bound[1]=="Net Total (MYR)":
          # draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
          # print(bound[0])
          # image.save("temp.jpg")
    return image
# draw_boxes(im, bounds)
def concatenate_list_data(list):
    result= ''
    for element in list:
        result = result +str(element)
    return result


for s in filenames:
    # s = "Invoice0.jpg"
    filen = s.split(".")[0]
    print(filen)
    im = PIL.Image.open(s).convert('RGB')
    # Doing OCR. Get bounding boxes.
    bounds = reader.readtext(s)
    im = PIL.Image.open(s).convert('RGB')
    df = pd.DataFrame()
    CoordinatesValue = []
    for i in bounds:
        Coordinates =[]
        CoordinatesValue=[]
        temp_df = pd.DataFrame()
        Coordinates.append(concatenate_list_data(i[0]).replace("][",",").replace("[","").replace("]","").replace(" ",""))
        print(i[1])
        CoordinatesValue.append(i[1])
        temp_df = DataFrame(zip(Coordinates,CoordinatesValue),columns = ['Coordinates', 'Value'])
        print(temp_df)
        df = df.append(temp_df)
    # print(item[0])
    combine_lambda = lambda x: '{},{}'.format(x.Coordinates, x.Value)
    df['Result'] = df.apply(combine_lambda, axis = 1)
    dfnew= df['Result']
    dfnew = dfnew[0].str.split(',', expand=True)
    dfnew.to_csv(str(filen)+".csv", index=False ,header=False )



