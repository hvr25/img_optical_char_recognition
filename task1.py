
        #  opencv-python   -   4.5.5.64


"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    features = enrollment(characters)

    img = test_img.copy()
    labels = detection(img)
    
    output = recognition(test_img,features,labels)

    return output


    # raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    sift = cv2.SIFT_create()
    features = {}
    for i in range(len(characters)):
        kp, des = sift.detectAndCompute(characters[i][1],None)
        features[characters[i][0]] = des
    return features

    # raise NotImplementedError

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    
    img_h = len(test_img)
    img_w = len(test_img[0])
    threshold = 150
    # print(np.average(test_img))
    for u,el in enumerate(test_img):
        for v,ele in enumerate(el):
            if ele<threshold:
                test_img[u][v] = 1
            else:
                test_img[u][v] = 0

    temp_lbl = test_img.tolist()
    label_val = 1
    labels = {}
    
    for i in range(img_h):
        for j in range(img_w):
            if temp_lbl[i][j] > 0:
                if temp_lbl[i - 1][j - 1]==0 and temp_lbl[i][j - 1]==0 and temp_lbl[i - 1][j]==0 and temp_lbl[i - 1][j + 1]==0:
                    temp_lbl[i][j] = label_val
                    labels[label_val] = [label_val]
                    label_val += 1
                else:
                    nbr_label = [temp_lbl[i][j - 1], temp_lbl[i - 1][j - 1], temp_lbl[i - 1][j], temp_lbl[i - 1][j + 1]]
                    nbr_label = [int(x) for x in nbr_label if x > 0]
                    min_label = min(nbr_label)
                    temp_lbl[i][j] = min_label

                    if min_label not in labels:
                        for x in labels:
                            if min_label in labels[x]:
                                min_label = x
                    for x in nbr_label:
                        if x not in labels[min_label]:
                            labels[min_label].append(x)
                            if x in labels:
                                for y in labels[x]:
                                    if y not in labels[min_label]:
                                        labels[min_label].append(y)
                                del labels[x]

    cnt = 1
    final_label={}
    for x in labels:
        final_label[cnt] = labels[x]
        cnt+=1
    for i in range(img_h):
        for j in range(img_w):
            if temp_lbl[i][j]>0:
                for x in final_label:
                        if temp_lbl[i][j] in final_label[x]:
                            temp_lbl[i][j] = x
            # test_img[i][j] = temp_lbl[i][j]

    for x in final_label:
        final_label[x]=[img_h,img_w,0,0]

    for i in range(img_h):
        for j in range(img_w):
            if temp_lbl[i][j]>0:
                if i < final_label[temp_lbl[i][j]][0]:
                    final_label[temp_lbl[i][j]][0] = i
                if i > final_label[temp_lbl[i][j]][2]:
                    final_label[temp_lbl[i][j]][2] = i
                if j < final_label[temp_lbl[i][j]][1]:
                    final_label[temp_lbl[i][j]][1] = j
                if j > final_label[temp_lbl[i][j]][3]:
                    final_label[temp_lbl[i][j]][3] = j

    for lbl in final_label:
        x = final_label[lbl][1]
        y = final_label[lbl][0]
        h = final_label[lbl][2] - final_label[lbl][0] +1
        w = final_label[lbl][3] - final_label[lbl][1] +1
        final_label[lbl] = [x,y,w,h]
   
    return final_label

    # raise NotImplementedError

def recognition(test_img,features,labels):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    sift = cv2.SIFT_create()
    output_list=[]
    for x in labels:
        clas = {"bbox":labels[x],"name":"UNKNOWN"}
        kp, des = sift.detectAndCompute(test_img[labels[x][1]-3:labels[x][1]+labels[x][3]+3,labels[x][0]-3:labels[x][0]+labels[x][2]+3],None)
        for ft in features:
            match_cnt = 0
            if des is not None and features[ft] is not None:
                for i in range(len(features[ft])):
                    for j in range(len(des)):
                        ssd = np.sum(np.square(np.subtract(features[ft][i],des[j])))
                        if ssd < 100000:
                            match_cnt+=1                
            if match_cnt > 4:
                clas["name"] = ft
            if des is not None and labels[x][2]<=6 and labels[x][3]<=6:
                clas["name"] = 'dot'
        output_list.append(clas)
    return output_list

    # raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
