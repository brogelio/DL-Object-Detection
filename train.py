import gdown

import gzip
import shutil

import tarfile

from collections import defaultdict
import os
import csv

from xml.etree.ElementTree import parse, Element, SubElement, ElementTree

from detecto import core, utils, visualize


url = "https://drive.google.com/file/d/1smM0MTR5EohrsV1XOYOclgEQccNnRxAX/view?usp=sharing"
output = 'drinks.tar.gz'
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# Decompress .gz file
with gzip.open('drinks.tar.gz', 'rb') as f_in:
    with open('drinks.tar', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Decompress .tar file
my_tar = tarfile.open('drinks.tar')
my_tar.extractall('./drinks')
my_tar.close()

# csv to xml for test
save_root2 = "test"

if not os.path.exists(save_root2):
    os.mkdir(save_root2)


def write_xml(folder, filename, bbox_list):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = './images' + filename
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = bbox_list[0]

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = e_width
    SubElement(size, 'height').text = e_height
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for entry in bbox_list:
        e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = entry

        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = e_class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = e_xmin
        SubElement(bbox, 'ymin').text = e_ymin
        SubElement(bbox, 'xmax').text = e_xmax
        SubElement(bbox, 'ymax').text = e_ymax

    tree = ElementTree(root)

    xml_filename = os.path.join('.', folder, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename)


entries_by_filename = defaultdict(list)

with open('drinks/drinks/labels_test.csv', 'r', encoding='utf-8') as f_input_csv:
    csv_input = csv.reader(f_input_csv)
    header = next(csv_input)

    for row in csv_input:
        filename, xmin, xmax, ymin, ymax, class_name = row
        if class_name == '1':
            class_name = "Water"
        elif class_name == '2':
            class_name = "Soda"
        elif class_name == '3':
            class_name = "Juice"
        width = '640'
        height = '480'
        entries_by_filename[filename].append([filename, width, height, class_name, xmin, ymin, xmax, ymax])

for filename, entries in entries_by_filename.items():
    print(filename, len(entries))
    write_xml(save_root2, filename, entries)

# csv to xml for train
save_root2 = "train"

if not os.path.exists(save_root2):
    os.mkdir(save_root2)

entries_by_filename = defaultdict(list)

with open('drinks/drinks/labels_train.csv', 'r', encoding='utf-8') as f_input_csv:
    csv_input = csv.reader(f_input_csv)
    header = next(csv_input)

    for row in csv_input:
        filename, xmin, xmax, ymin, ymax, class_name = row
        if class_name == '1':
            class_name = "Water"
        elif class_name == '2':
            class_name = "Soda"
        elif class_name == '3':
            class_name = "Juice"
        width = '640'
        height = '480'
        entries_by_filename[filename].append([filename, width, height, class_name, xmin, ymin, xmax, ymax])

for filename, entries in entries_by_filename.items():
    print(filename, len(entries))
    write_xml(save_root2, filename, entries)

# move files
train_files = []
test_files = []
for i in range(1001):
    if i < 10:
        file_name = '000000' + str(i) + '.jpg'
    if 100 > i > 9:
        file_name = '00000' + str(i) + '.jpg'
    if 1000 > i > 99:
        file_name = '0000' + str(i) + '.jpg'
    if i == 1000:
        file_name = '0001000.jpg'
    train_files.append(file_name)
for i in range(51):
    i += 10000
    file_name = '00' + str(i) + '.jpg'
    test_files.append(file_name)

for f in train_files:
    shutil.copy('drinks/drinks/%s' %f, 'train')
for g in test_files:
    shutil.copy('drinks/drinks/%s' %g, 'test')

dataset = core.Dataset('train')
model = core.Model(['Water', 'Soda', 'Juice'])

model.fit(dataset, verbose=True)
model.save('model_weights_v2.pth')






