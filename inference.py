import os
import os
print(os.getcwd())
# os.path.dirname(os.path.abspath("__file__"))
path = '/Volumes/Extreme SSD/MLWork/DocAI/PICK-pytorch'
os.chdir(path)
# os.chdir('../')
# path = '/Users/neerajyadav/Documents/pycv/PICK-pytorch/'

"""Convert files of a selected directory in jpg format"""
import converter
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
import csv
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
import converter
import shutil, os
device = torch.device(f'cuda:{args.gpu}' if -1 != -1 else 'cpu') ### setting value of gpu to -1 to run inference 
savedCheckpiont = 'saved/models/PICK_Default/test_999/model_best.pth'
checkpoint = torch.load(savedCheckpiont, map_location=device)
config = checkpoint['config']
state_dict = checkpoint['state_dict']
monitor_best = checkpoint['monitor_best']
print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(savedCheckpiont, monitor_best))
# prepare model for testing
pick_model = config.init_obj('model_arch', pick_arch_module)
pick_model = pick_model.to(device)
pick_model.load_state_dict(state_dict)
pick_model.eval()
## pick ocr transcript file and image in below folders
out_img_path = "test_img/"
out_box_path = "test_boxes_and_transcripts/"

def generateTranscript():
    ### convert image into transcript file 
    """Select jpg files and convert into transcript files"""
    filenames = glob.glob("../TestImage/*.jpg")
    filenamesj = glob.glob("../TestImage/*.jpeg")
    filenames = filenames + filenamesj
    filenames.sort()

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
    #         print(i[1])
            CoordinatesValue.append(i[1])
            temp_df = DataFrame(zip(Coordinates,CoordinatesValue),columns = ['Coordinates', 'Value'])
    #         print(temp_df)
            df = df.append(temp_df)
        # print(item[0])
        combine_lambda = lambda x: '{},{}'.format(x.Coordinates, x.Value)
        df['Result'] = df.apply(combine_lambda, axis = 1)
        dfnew= df['Result']
        dfnew = dfnew[0].str.split(',', expand=True)
        dfnew.insert(0,'name_of_column','')
        dfnew['name_of_column'] = 1
    #     dfnew.to_csv(str(filen)+".tsv",  sep = ',',index=False ,header=False )
        dfnew.to_csv(str(filen)+".tsv",sep = ',',index=False,header=False, quotechar='',escapechar='\\',quoting=csv.QUOTE_NONE, )

    ### copy file from source folder to destination folder ###
    for f in filenames:
        shutil.copy(f, 'test_img/')
    filetsv = glob.glob("/Volumes/Extreme SSD/MLWork/DocAI/TestImage/*.tsv")
    for f in filetsv:
        shutil.copy(f, 'test_boxes_and_transcripts/')

def runInference():
    ### inference code ###
    # setup dataset and data_loader instances
    batch_size_val=1
    test_dataset = PICKDataset(boxes_and_transcripts_folder=out_box_path,
                                images_folder=out_img_path,
                                resized_image_size=(480, 960),
                                ignore_error=False,
                                training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False,
                                    num_workers=0, collate_fn=BatchCollateFn(training=False)) ## have changed the number of workers to zero

    # setup output path
    output_folder = 'output'
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for step_idx, input_data_item in enumerate(test_data_loader):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)
            # For easier debug.
            image_names = input_data_item["filenames"]
            # print('image names')
            # print(image_names)
            output = pick_model(**input_data_item)
            # print(output)
            logits = output['logits']  # (B, N*T, out_dim)
            # print(logits)
            new_mask = output['new_mask']
            # print(new_mask)
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            # print('best_paths')
            # print(best_paths)
            predicted_tags = []
            for path, score in best_paths:
                # print(path,score)
                predicted_tags.append(path)
            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)
            # print(decoded_texts_list)
            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])
                entities = []  # exists one to many case
                # print(spans)
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                    text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)
                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                # print(entities)
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))
                        print(item['entity_name'],item['text'])
        # dir = 'path/to/dir'
        try:
            dir = out_img_path
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        except:
            pass
        try:
            dir = out_box_path
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        except:
            pass
    

    
     


