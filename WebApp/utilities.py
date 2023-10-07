import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import cv2
from collections import Counter

from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('ResNet50_FineTuning_SportsClassification.h5')

SportsDictionary = {0: 'air hockey',
                    1: 'ampute football',
                    2: 'archery',
                    3: 'arm wrestling',
                    4: 'axe throwing',
                    5: 'balance beam',
                    6: 'barell racing',
                    7: 'baseball',
                    8: 'basketball',
                    9: 'baton twirling',
                    10: 'bike polo',
                    11: 'billiards',
                    12: 'bmx',
                    13: 'bobsled',
                    14: 'bowling',
                    15: 'boxing',
                    16: 'bull riding',
                    17: 'bungee jumping',
                    18: 'canoe slamon',
                    19: 'cheerleading',
                    20: 'chuckwagon racing',
                    21: 'cricket',
                    22: 'croquet',
                    23: 'curling',
                    24: 'disc golf',
                    25: 'fencing',
                    26: 'field hockey',
                    27: 'figure skating men',
                    28: 'figure skating pairs',
                    29: 'figure skating women',
                    30: 'fly fishing',
                    31: 'football',
                    32: 'formula 1 racing',
                    33: 'frisbee',
                    34: 'gaga',
                    35: 'giant slalom',
                    36: 'golf',
                    37: 'hammer throw',
                    38: 'hang gliding',
                    39: 'harness racing',
                    40: 'high jump',
                    41: 'hockey',
                    42: 'horse jumping',
                    43: 'horse racing',
                    44: 'horseshoe pitching',
                    45: 'hurdles',
                    46: 'hydroplane racing',
                    47: 'ice climbing',
                    48: 'ice yachting',
                    49: 'jai alai',
                    50: 'javelin',
                    51: 'jousting',
                    52: 'judo',
                    53: 'lacrosse',
                    54: 'log rolling',
                    55: 'luge',
                    56: 'motorcycle racing',
                    57: 'mushing',
                    58: 'nascar racing',
                    59: 'olympic wrestling',
                    60: 'parallel bar',
                    61: 'pole climbing',
                    62: 'pole dancing',
                    63: 'pole vault',
                    64: 'polo',
                    65: 'pommel horse',
                    66: 'rings',
                    67: 'rock climbing',
                    68: 'roller derby',
                    69: 'rollerblade racing',
                    70: 'rowing',
                    71: 'rugby',
                    72: 'sailboat racing',
                    73: 'shot put',
                    74: 'shuffleboard',
                    75: 'sidecar racing',
                    76: 'ski jumping',
                    77: 'sky surfing',
                    78: 'skydiving',
                    79: 'snow boarding',
                    80: 'snowmobile racing',
                    81: 'speed skating',
                    82: 'steer wrestling',
                    83: 'sumo wrestling',
                    84: 'surfing',
                    85: 'swimming',
                    86: 'table tennis',
                    87: 'tennis',
                    88: 'track bicycle',
                    89: 'trapeze',
                    90: 'tug of war',
                    91: 'ultimate',
                    92: 'uneven bars',
                    93: 'volleyball',
                    94: 'water cycling',
                    95: 'water polo',
                    96: 'weightlifting',
                    97: 'wheelchair basketball',
                    98: 'wheelchair racing',
                    99: 'wingsuit flying'}


def takeaArgMax (npArray):
    '''
    This function takes a numpy array and returns the index of the highest value in it (Which is the label)
    '''

    label = np.argmax(npArray)

    return label

def ConvertToSportName (label,dictionary):
    TheSportName = dictionary[label]

    return TheSportName

def PredictOneImage (Image_path):
    '''
    This function takes the image path and return the predicted label
    '''
    img = image.load_img(Image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch)

    label = takeaArgMax (predictions)


    return label

def extract_frames (Video_path):
    '''
    This function takes the video path and extract frames from it and save it in  a folder called "frames"
    '''
    os.mkdir('frames')
    # Open the video file
    cap = cv2.VideoCapture(Video_path)

    # Check if video opened successfully
    if not cap.isOpened(): 
        print("Error opening video file")

    # Read until video is completed
    frame_id = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            cv2.imwrite('frames/frame%d.jpg' % frame_id, frame)
            frame_id += 1

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def predictFrames (Video_path, dictionary):
    
    extract_frames (Video_path)

    labels = []
    for file_name in  os.listdir('frames'):


        path = 'frames/'+file_name
        labels.append(PredictOneImage (path))
    
    counter = Counter(labels)

    most_common_element = counter.most_common(1)[0][0]

    TheSportName = ConvertToSportName (most_common_element,dictionary)

    shutil.rmtree('frames')
    
    return TheSportName