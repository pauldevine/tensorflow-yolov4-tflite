import glob
import os
import tensorflow as tf
import elementpath
from xml.etree import ElementTree as ET
from absl import app, flags, logging
from absl.flags import FLAGS

OBJECT_TYPES=['metridium', 'fish']

FLAGS = flags.FLAGS

flags.DEFINE_string('input', '~/projects/tensorflow-yolov4-tflite/data/mare_hand_annotated/xml/*.xml', 'path to input xml files in rectlabel format')
flags.DEFINE_string('output', '~/projects/tensorflow-yolov4-tflite/data/mare_hand_annotated/train_val_test', 'path to output directory') 


def _parse_function(filename):
  
  #tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
  #print("inside parse filename: {}".format(filename))

  #raw_xml = tf.gfile.GFile(filename, "rb").read()
 
  with tf.io.gfile.GFile(filename, "rb") as file:
    contents = file.read().decode("utf-8")
    #print(contents)
    file.seek(0)
    tree = ET.parse(filename)
  root = tree.getroot()
  folder = root.findall('./folder')[0].text
  
  #/Users/pdevine/projects/ILSVRC/Annotations/CLS-LOC/train/n07697537/n07697537_12752.xml
  #/Users/pdevine/projects/ILSVRC/Data/CLS-LOC/train/n07697537/n07697537_12752.JPEG
  base_filename = os.path.basename(os.path.splitext(filename)[0])
  img_filename = base_filename +'.jpg'
  txt_filename = base_filename + '.txt'

  img_filename = img_filename.replace("Annotations", "Data")

  print('folder:', folder, 'filename:', filename, 'img_filename:', img_filename, 'txt_filename:', txt_filename)

  imageid = root.find('.filename').text

  features = {"img_filename": img_filename,
              "txt_filename": txt_filename,
              "width": root.find('./size/width').text,
              "height": root.find('./size/height').text,
              "depth": root.find('./size/depth').text}
           
  labels = {"ImageId": imageid,
           "PredictionString": []}
  #print('width: {} height: {} depth: {} segmented: {}'.format(features['width'], features['height'], features['depth'], features['segmented']))

  for child in root.findall('./object'):
    box = {
      "name": child.find('name').text,
      "x_min": child.find('bndbox/xmin').text,
      "y_min": child.find('bndbox/ymin').text,
      "x_max": child.find('bndbox/xmax').text,
      "y_max": child.find('bndbox/ymax').text }
    labels["PredictionString"].append(box)

  features['labels'] = labels
  return features


#outfile format
#Row format: image_file_path box1 box2 ... boxN;
#Box format: x_min,y_min,x_max,y_max,class_id (no space).
#/home/pdevine/keras-yolo3-v2/open-images-dataset/train/8d6e7c7f05dd136f.jpg 0,0,682,1023,148

def main(argv): 
  print('Processing XML files from RectLabel')
  print('Input files frm:',FLAGS.input)
  print('Saving files to:',FLAGS.output)
  for filename in glob.glob(os.path.expanduser(FLAGS.input)):
      print(filename)
      features = _parse_function(filename)
      outfilename = FLAGS.output + '/' + features['txt_filename']
      outfilename = os.path.expanduser(outfilename)
      with open(outfilename, 'w') as outfile:
        print(features['labels']['PredictionString'])
        boxes = features['labels']['PredictionString']
      
        for bbox in boxes:
          outfile.write(bbox['x_min'] + ',')
          outfile.write(bbox['y_min'] + ',')

          outfile.write(bbox['x_max'] + ',')
          outfile.write(bbox['y_max'] + ',')
          obj_id = OBJECT_TYPES.index(bbox['name'])
          outfile.write(str(obj_id) + ' ')
        outfile.write('\n')
        # with open(filename, 'rb') as readfile:
        #     basename = os.path.splitext(os.path.basename(filename))[0]
        #     img_path = '/home/pdevine/mare_video/test_1/' + basename + '.jpg'
        #     print('\n' + img_path, end=' ')
        #     line = readfile.readline()

        #     while line:
        #       print(line.strip(), end=' ')
      #       line = readfile.readline()

if __name__ == '__main__':
  app.run(main)