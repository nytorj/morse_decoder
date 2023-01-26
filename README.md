Morse Decoder inspired by the PicoCTF 2022 challange(morse-code).
I've been watching the PicoCTF challenge on John's Hammond youtube channel,
and he, after solving it with an online tool, suggested that we tried out
ourselves and try to solve it in python...

How it worsk:

input file > librosa audio library > batch split > matplotlib > cv2 > crop > pixelscan to string > string to char.

After extracting the audio with librosa, we can split the data array into regular chunks = max number of samples desired. We make sure
that we split the array at the right place, between two letter or 2 phrases.
We map out the pixels values and convert them to 0s and 1s
We then convet the new string into a dash dot string, and decode it.

**We have to batch the array, because we don't want to loose resolution on a fixed image size on a longer message.**
**Batching the array on the right spot was trick. You gotta make sure you split the array between words and phrases**
**Trick diference btween how matplotlib libray exports the plot figure inside jupyter notebook versus outside jupyter notebook.**(this was a pain in the ass)

