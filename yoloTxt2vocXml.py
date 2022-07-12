#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:MXY
# yolo的标签转换为labelImg形式的xml标签

import shutil
from xml.dom.minidom import Document
import os
import cv2

# ------------------------------ 修改内容 ----------------------------------
img_path = r'images'  # 图片数据文件夹
txt_path = r'labels'  # txt文件夹
xml_path = r'xml'  # xml文件夹
Format = '.png'
dict = {'0': "003"}  # 字典对类型进行转换，标签类别列表。


# -------------------------------------------------------------------------


def makexml(txtPath, xmlPath, picPath, format='.png'):  # 读取txt路径，xml保存路径，数据集图片所在路径
    if os.path.exists(xmlPath):
        shutil.rmtree(xmlPath)
    os.makedirs(xmlPath)
    files = os.listdir(txtPath)
    lenfiles = len(files)
    percent, num = 10, 0
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(os.path.join(txtPath, name))
        txtList = txtFile.readlines()
        img = cv2.imread(os.path.join(picPath, name[0:-4] + format))
        Pheight, Pwidth, Pdepth = img.shape
        for txt in txtList:
            oneline = txt.strip().split(" ")

            folder = xmlBuilder.createElement("folder")  # folder标签
            folderContent = xmlBuilder.createTextNode("VOC2007")
            folder.appendChild(folderContent)
            annotation.appendChild(folder)

            filename = xmlBuilder.createElement("filename")  # filename标签
            filenameContent = xmlBuilder.createTextNode(name[0:-4] + format)
            filename.appendChild(filenameContent)
            annotation.appendChild(filename)

            size = xmlBuilder.createElement("size")  # size标签
            width = xmlBuilder.createElement("width")  # size子标签width
            widthContent = xmlBuilder.createTextNode(str(Pwidth))
            width.appendChild(widthContent)
            size.appendChild(width)
            height = xmlBuilder.createElement("height")  # size子标签height
            heightContent = xmlBuilder.createTextNode(str(Pheight))
            height.appendChild(heightContent)
            size.appendChild(height)
            depth = xmlBuilder.createElement("depth")  # size子标签depth
            depthContent = xmlBuilder.createTextNode(str(Pdepth))
            depth.appendChild(depthContent)
            size.appendChild(depth)
            annotation.appendChild(size)

            object = xmlBuilder.createElement("object")
            picname = xmlBuilder.createElement("name")
            nameContent = xmlBuilder.createTextNode(dict[oneline[0]])
            picname.appendChild(nameContent)
            object.appendChild(picname)
            pose = xmlBuilder.createElement("pose")
            poseContent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(poseContent)
            object.appendChild(pose)
            truncated = xmlBuilder.createElement("truncated")
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)
            difficult = xmlBuilder.createElement("difficult")
            difficultContent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultContent)
            object.appendChild(difficult)
            bndbox = xmlBuilder.createElement("bndbox")
            xmin = xmlBuilder.createElement("xmin")
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)
            ymin = xmlBuilder.createElement("ymin")
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)
            xmax = xmlBuilder.createElement("xmax")
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)
            ymax = xmlBuilder.createElement("ymax")
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)
            object.appendChild(bndbox)

            annotation.appendChild(object)

        f = open(os.path.join(xmlPath, name[0:-4] + '.xml'), 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()
        num += 1
        if num >= lenfiles * percent / 100:
            print('Finished %s%%.' % percent)
            percent += 10


if __name__ == '__main__':
    img_path = r'C:\Users\zk\Desktop\yolov5-v6.1\yolov5-master\test_20220510_024-039\images'
    txt_path = r'C:\Users\zk\Desktop\yolov5-v6.1\yolov5-master\runs\detect\003\labels'
    xml_path = r'C:\Users\zk\Desktop\yolov5-v6.1\yolov5-master\runs\detect\003\xml'

    makexml(txt_path,
            xml_path,
            img_path,
            format=Format)
