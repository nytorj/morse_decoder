#!/usr/bin/env python3
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv2
import os
import pathlib


def delete_images(img_path, extension):
        for file in os.listdir(img_path):
            if file.endswith(extension):
                os.remove(os.path.join(img_path, file))


def load_audio(path):
    y, sr = librosa.load(path)
    return y, sr

def plot_audio(y, name):
    plt.figure(figsize=(15, 17))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y)
    plt.title(name)
    plt.savefig(name, edgecolor='white',  dpi=72)


def load_image(path):
    img = cv2.imread(path)
    return img

def save_image(img, path):
    cv2.imwrite(path, img)

#fuction to batch split an array into sclices of given size
def batch_split(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]

#function to validade if an array has the first 2 elements are equal to 0 and the last 2 elements are equal to 0
def is_zero(arr):
    return np.all(arr[:79000] == 0) and np.all(arr[-79000:] == 0)

def detect_edges(img):
    #convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #return the edged image
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)


    return edges

def detect_lines(img, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=100)

    return lines

def crop_image(image, lines):
    lines_list = []

    for points in lines:
        x1, y1, x2, y2 = points[0]
        lines_list.append([x1, y1, x2, y2])
    xarr = []
    yarr = []

    for line in lines_list:
        xarr.append(line[0])
        yarr.append(line[1])
        xarr.append(line[2])
        yarr.append(line[3])

        x1 = np.array(xarr).min()+40
        x2 = np.array(xarr).max()-20#!!
        y1 = np.array(yarr).min()+50
        y2 = np.array(yarr).min()+52
        crop_img = image[int(y1):int(y2),int(x1):int(x2)]

    return crop_img


def image_to_string(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def find_max_samples(arr):

    max_samples = 400000
    jump_size = 8000
    found = False
    while found == False:

        arrs = batch_split(y, max_samples)
        for a in arrs:
            if(is_zero(a)):
                continue
            else:
                max_samples+=jump_size
                continue
        found = True

    print(max_samples)
    return(arrs, max_samples)

#function to filter array for zeros at the begining until a non zero element is found
def filter_zeros(arr):
    for i in range(len(arr)):
        if(arr[i] == 0):
            continue
        else:
            return arr[i:]
    return arr
#function to filter array for zeros at the end until a non zero element is found
def filter_zeros_end(arr):
    for i in range(len(arr)-1, -1, -1):
        if(arr[i] == 0):
            continue
        else:
            return arr[:i]
    return arr
#function to filter array for non zeros at the end until a zero element is found
def filter_non_zeros_end(arr):
    for i in range(len(arr)-1):
        if(arr[-i]!= 0):
            continue
        else:
            return arr[:-i+1]
#function to filter array for non zeros at the begining until a zero element is found
def filter_non_zeros_begin(arr):
    for i in range(len(arr)-1):
        if(arr[i]!= 0):
            continue
        else:
            return arr[i:]

def treat_array(arr, padding):
    arr = filter_zeros(arr)
    arr = filter_zeros_end(arr)
    arr = np.pad(arr, (padding, padding), 'constant')
    return arr

#########################################################
#compare letter by letter of the string to the
#morse code dictionary to decode the message
#########################################################
def decode_morse_string(letter):

    MORSE_CODE_DICT = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----",
        ", ": "--..--",
        ".": ".-.-.-",
        "?": "..--..",
        "/": "-..-.",
        "-": "-....-",
        "(": "-.--.",
        ")": "-.--.-",
        "'": ".----.",
        "!": "-...",
        "@": ".--.-.",
        "#": "-.-..-",
        "$": "..--..",
        "%": ".-..-.",
        "&": "...-.",
        "+": ".-.-.",
    }
    match = ""
    for ltr in MORSE_CODE_DICT:
        if MORSE_CODE_DICT[ltr] == letter:
            match+=ltr
            break
    return match


path = input("Enter the path of the audio file:>")

image_name = 'plot.png'
cropped_image_name = 'cropped_plot.png'

y, sr = load_audio(path)
y_padding = int(sr/20)
y = treat_array(y, y_padding)

slices, max_samples = find_max_samples(y)

n=0
dash_dot_string = ""
decoded_string = ""
dash_dot_letter = ""
for slice in slices:
    n+=1
    if(len(slice)<max_samples):
        slice = np.pad(slice, (0, max_samples-len(slice)), 'constant')

    plot_audio(slice, f"slice_{n}_{image_name}")
    image = load_image(f"slice_{n}_{image_name}")

    #print(image)
    edges = detect_edges(image) # detect
    lines = detect_lines(image, edges) # detect
    crop_img = crop_image(image, lines) # crop
    #clear image variable
    #image = []
    save_image(crop_img, f"slice_{n}_{cropped_image_name}")


    im = Image.open(f"slice_{n}_{cropped_image_name}")
    pix = im.load()
    width, height = im.size
    image_pixels = []

    for i in range(0, width):
        x, y, z = pix[i, 1]
        image_pixels.append(x)

    #print(image_pixels)
    string_image = ""
    for char in image_pixels:
        if(char > 100):
            char = 0
        else:
            char = 1
        string_image+=str(char)

    zeros = 0
    ones = 0
    letter_added = True

    for nbr in string_image:

        if nbr == "0":
            zeros+=1

            if (ones != 0):
                if (ones < 6):
                    dash_dot_string += '.'
                    dash_dot_letter += '.'
                else:
                    dash_dot_string += '-'
                    dash_dot_letter += '-'
                ones = 0

        elif nbr == "1":
            ones+=1
            letter_added = False
            if zeros != 0:
                if zeros < 6:
                    pass
                else:

                    if(6 < zeros < 10):
                        dash_dot_string+=" "
                        if(letter_added == False and dash_dot_letter!=""):
                            decoded_string+=decode_morse_string(dash_dot_letter)
                            dash_dot_letter= ""
                            decoded_string+=" "
                            letter_added = True
                    else:
                        decoded_string+=" "
                        dash_dot_string+="  "
                zeros=0
            else:
                continue


def dash_dot_space_3xspace_string_decoder(string):
    letter_appended = False
    letter = ""
    letters = []

    for char in string:
        if char != " ":
            letter_appended = False
            letter += char
        else:
            if(letter_appended == False):
                letters.append(letter)
                letters.append(" ")
                letter = ""
            letter_appended = True
    return letters


letters = dash_dot_space_3xspace_string_decoder(dash_dot_string+" ")


final_message = ""
for let in letters:
    to_append_letter = decode_morse_string(let)
    final_message+=to_append_letter


print(final_message)


