# Pic2Char

### About
A tiny Python program to convert picture to chars, get hog features from picture's areas and using char with most similiar hog feathre to represident it, finally convert a picture to a list of chars.

### Requirement
* Python 2.7
* OpenCV 3
* Numpy
* Movie-py

### Usage
```python
python pic2char.py <source> <max(width, height)> <target_file> [<channels> [<to_txt_file> [<trim_txt_file>]]]
```
   * source: source file, can be picture, gif or video file (gif and video consume huge amount time)  
   * max(width, height): max width or height of output file, choose the longer one
   * target_file: The file you want to export, you can export to any file if you set `to_txt_file` to 1, but you need to choose right fiel type otherwise, picture to picture, gif to gif, video to video
   * channels: the channels use to extract features, 3 or 1.
    * to_txt_file: **ONLY WORKS IN PICTURE**, 1 means output to txt file, otherwise to pictures
    * trim_txt_file: **ONLY WORKS IN PICTURE**, 1 means skip blank in line end

### Example
Sample Picture (Get from [pixabay](https://pixabay.com/zh/研究员-晚礼服-inkscape中-矢量-企鹅-动物-1625959/)):
  ![Sample][1]

* To 3 Channel picture:
  ![3Channel][2]

* To 1 Channel picture:
  ![1Channel][3]

* To TXT File:
  ![Txt][4]
  (The origin file is located in [txt.txt](./readme/txt.txt))

### Addition
You can edit some params defined at the top of [pic2char.py](./pic2char.py), the area's width or height, the chars to use, the gray values.

[1]: readme/sample.png
[2]: readme/3channel.png
[3]: readme/1channel.png
[4]: readme/txt.png
