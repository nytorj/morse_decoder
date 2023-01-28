#!/usr/bin/env python3

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv2
import os
import pathlib
import time

#global variables for data padronizaiton
min_number_of_zeros = 2016
max_samples = 400000
starting_samples = 400000
image_name = 'plot.png'
cropped_image_name = 'cropped_plot.png'
#not in use right now but the goal is to delete *plot.png after decoding
def delete_images(img_path, extension):
        for file in os.listdir(img_path):
            if file.endswith(extension):
                os.remove(os.path.join(img_path, file))
#did choose not to resample the audio file - did that kinda manually?
def load_audio(path):
    y, sr = librosa.load(path)
    return y, sr
#important settings for cropping purposes
def plot_audio(y, name):
    plt.figure(figsize=(15, 17))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y)
    plt.title(name)
    plt.savefig(name, edgecolor='white',  dpi=72)
    plt.close()
#cv2 load image
def load_image(path):
    img = cv2.imread(path)
    return img
#cv2 save image 
def save_image(img, path):
    cv2.imwrite(path, img)

#fuction to batch split an array into sclices of given size
def batch_split(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]

#function to validade if an array has the first 2 elements are equal to 0 and the last 2 elements are equal to 0
def is_zero(arr,zero_length):
    return np.all(arr[:zero_length] == 0) and np.all(arr[-zero_length:] == 0)
#cv2 detect edges
def detect_edges(img):
    #convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200, apertureSize=3)


    #return the edged image
    return edges
#detect lines for cropping a thin slice of the image
def detect_lines(img, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=100)

    return lines
#crop img for pixel by pixel analisys
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
#return the maximum number of zeros in a sequence in the array
#so we can check amount of data per samplerate more zeros, less data
#needed to padronization of spaces of dash dot, letters, phrases.
def find_max_sum(arr):
    z = []
    zeros = np.array(z)
    max_sum = 0
    for i in range(0, len(arr)):
        if arr[i] == 0:
            max_sum += 1
        else:
            zeros = np.append(zeros, max_sum) if max_sum!= 0 else zeros
            max_sum = 0
    return zeros.max()
#return the minimum number of zeros in a sequence in the array
#so we can add zeros if necessary. 
def find_min_sum(arr):
    z = []
    zeros = np.array(z)
    max_sum = 0
    for i in range(0, len(arr)):
        if arr[i] == 0:
            max_sum += 1
        else:
            zeros = np.append(zeros, max_sum) if max_sum!= 0 else zeros
            max_sum = 0
    return zeros.min()


#invert image color
def image_to_string(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#used to find the exact location to bacth the data array, if data array lenght is greater then max_sample per plot variable
def find_max_samples(arr):
    global max_samples
    global starting_samples
    
    farr = filter_zeros(arr)
    ffarr = filter_zeros_end(farr)
        
    max_zeros = find_max_sum(ffarr)
    
    jump_size = 100#-------------------------------------------------------------------important for cropping purposes
    found = 0
    done = False
    rounds = (len(arr)/max_samples)
    
    while not done:
        arrs = batch_split(arr, max_samples)
        for a in arrs:
            if(len(a)<starting_samples):
                a = np.pad(a, (0, starting_samples-len(a)), 'constant')
            
            if(is_zero(a, int(max_zeros/4))==False):#---------------------------------important for results
                continue            
            else:
                found += 1
        if(found > rounds):
            done = True
        else:
            found = 0
            max_samples-=jump_size

    #print(max_samples)
    return(arrs)


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
#fucntion to treat data array for padronization
def treat_array(arr, padding):
    narr = []
    ret_arr = np.array(narr)
    arr = filter_zeros(arr)
    arr = filter_zeros_end(arr)
    global max_samples, starting_samples
    min_sample_zeros = find_min_sum(arr)
    if(min_sample_zeros < min_number_of_zeros):
        starting_samples = int(300*min_sample_zeros)#============================================
    starting_zero = 0
    ending_zero = 0

    for i in range(len(arr)-1):
        if(arr[i] == 0):
            if(starting_zero == 0):
                starting_zero = i
        else:
            if starting_zero!=0:
                ending_zero = i-1
                a_ar = arr[:starting_zero-1]
                zero_arr = arr[starting_zero:ending_zero]
                c_arr = arr[ending_zero+1:]

                if len(zero_arr) < min_number_of_zeros:
                    padding = min_number_of_zeros -len(zero_arr)
                    zero_arr = np.pad(zero_arr,(0,padding), 'constant')
                starting_zero = 0
                ending_zero = 0
                ret_arr=np.concatenate((a_ar,zero_arr,c_arr), axis=0)
            
    
    
    ret_arr = np.pad(arr, (padding, padding), 'constant')
    return ret_arr

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






def main():
    global max_samples, starting_samples
    path = input("Enter the path of the audio file:>")
    y, sr = load_audio(path)
    #pad the array begining and ending
    y_padding = int(sr/20)
    #padronizaiton of audio data array
    treated_y = treat_array(y, y_padding)
    #batch the audio array based on settings found on padronization
    slices = find_max_samples(treated_y)

    #start decoding loop variables
    n=0
    dash_dot_string = ""
    decoded_string = ""
    dash_dot_letter = ""
    #start decoding the batches
    for slc in slices:

        n+=1
        #trick to keep data resolution after batching
        slic = np.pad(slc, (0, starting_samples-len(slc)), 'constant')
        #plot spectogram for pixel t pixel analisys
        plot_audio(slic, f"slice_{n}_{image_name}")
        #here easiest way choosen. you can transfer image from memory to other functions instead of savnig to disk
        image = load_image(f"slice_{n}_{image_name}")
        #start of cv2 shenanagans
        edges = detect_edges(image) # detect
        lines = detect_lines(image, edges) # detect
        crop_img = crop_image(image, lines) # crop

        save_image(crop_img, f"cropped_slice_{n}_{cropped_image_name}")
        
        #start pixel by pixel analisys
        im = Image.open(f"cropped_slice_{n}_{cropped_image_name}")
        pix = im.load()
        width, height = im.size
        image_pixels = []

        for i in range(0, width):
            x, y, z = pix[i, 1]
            image_pixels.append(x)
        image_pixel = filter_zeros_end(image_pixels)

        #padronization continues
        string_image = ""
        min_zero = 10000000000
        zeros_sum = 0
        for char in image_pixel:
            if(char > 100):
                char = 0
                zeros_sum+=1
            else:
                char = 1
                if(zeros_sum!=0 and zeros_sum < min_zero):
                    min_zero = zeros_sum
                    zeros_sum = 0
                else:
                    zeros_sum = 0

            string_image+=str(char)


        #converting 0s and 1s to dash dot string and decoding it
        zeros=0
        ones=0
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

    #decode dash dot string
    letters = dash_dot_space_3xspace_string_decoder(dash_dot_string+" ")


    final_message = ""
    for let in letters:
        to_append_letter = decode_morse_string(let)
        final_message+=to_append_letter
    #reset globals
    min_number_of_zeros = 2016
    max_samples = 400000
    starting_samples = 400000
    print(final_message)


if __name__ == "__main__":
    main()
