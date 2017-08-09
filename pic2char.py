# coding=UTF-8

import cv2
import numpy as np
import sys
import locale
import shutil
from moviepy.editor import *

# HOG 方向维度
bin_n = 8

# 替换的cell大小
cell_size = 8

# 字体大小（scale之后）
font_size = 0.5

# 最终放大
out_size = 1

# 过程中放大（为了非衬线字体）
scale = 2

# 对于文本输出是否删除最后的空格
strip = False

# 字体（非衬线字体）
font = cv2.FONT_HERSHEY_SIMPLEX 

# 模拟字符
chars = ['-', '+', '=', 'B', 'O', '|', '\\', ' ', '/', 'T', '*', 'X', 'L', 'A', 'V', '<', '>', 'P', 'W', '@', 'J', 'M', '#', '&', 'U']
lux = None

gamma = 1

# 图片输出匹配灰度
gray = [0, 60, 120, 180, 240]
to_txt = False
max_dimension = 1200.0

def gamma_normalization(img, gamma):
    global lux
    if lux is None:
        lux = np.arange(0, 256, 1, dtype=np.uint8)
        lux = np.uint8(np.power(lux / 255.0, gamma) * 255)
    map_func = np.vectorize(lambda x: lux[x])
    dst = map_func(img)
    return dst


def hog(img):
    ys = cv2.filter2D(img, cv2.CV_32F, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    xs = cv2.filter2D(img, cv2.CV_32F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    mag, ang = cv2.cartToPolar(xs, ys)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    h, w = img.shape
    w /= cell_size
    h /= cell_size
    cells = np.zeros((w, h, bin_n), np.float32)
    for i in range(0, w):
        for j in range(0, h):
            mag_cell = mag[j * cell_size: j * cell_size + cell_size, i * cell_size: i * cell_size + cell_size]
            bin_cell = bins[j * cell_size: j * cell_size + cell_size, i * cell_size: i * cell_size + cell_size]
            cells[i, j] = np.bincount(bin_cell.ravel(), mag_cell.ravel(), bin_n) / np.square(cell_size)
    return cells


def distance(x, y):
    return np.sqrt(sum(np.square(x - y)))


def get_index(p, li):
    temp = map(distance, [p for i in range(len(li))], li)
    i = 0
    for k in range(1, len(temp)):
        if temp[k] < temp[i]:
            i = k
    return i


def get_similar(item, ori):
    x, y, d = item.shape
    result = np.zeros((x, y), np.uint8)
    for i in range(x):
        for j in range(y):
            result[i, j] = get_index(item[i, j], ori)
    return result

def strip_img(img, skip):
    x, y = img.shape
    x_start = 0
    x_end = x
    y_start = 0
    y_end = y
    for i in range(x):
        items = img[i, ]
        flag = True
        for item in items:
            if item not in skip:
                flag = False
        if flag:
            x_start += 1
        else:
            break
    for i in range(x - 1, -1, -1):
        items = img[i, ]
        flag = True
        for item in items:
            if item not in skip:
                flag = False
        if flag:
            x_end -= 1
        else:
            break
    for i in range(y):
        items = img[:, i]
        flag = True
        for item in items:
            if item not in skip:
                flag = False
        if flag:
            y_start += 1
        else:
            break
    for i in range(y - 1, -1, -1):
        items = img[:, i]
        flag = True
        for item in items:
            if item not in skip:
                flag = False
        if flag:
            y_end -= 1
        else:
            break
    result = np.zeros((x_end - x_start, y_end - y_start), np.uint8)
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            result[i - x_start , j - y_start] = img[i, j]
    return result

def process_single_img(img, items):
    print "strip" + str(strip)
    print "to_txt " + str(to_txt)
    print "gray " + str(gray)
    ox, oy = img.shape[0:2]
    rate = max_dimension / ox if ox >= oy else max_dimension / oy
    print "resize rate " + str(rate)
    img = cv2.resize(img, (0, 0), fx=rate, fy=rate)
    print "resize to " + str(img.shape)
    divide = cv2.split(img)
    img_result = []
    for i in range(len(divide)):
        origin = gamma_normalization(divide[i], gamma)
        print origin.dtype
        cells = hog(origin)
        result = get_similar(cells, items)
        print result
        x, y, d = cells.shape
        if strip:
            null = chars.index(' ')
            skip = range(null * len(gray), (null + 1) * len(gray))
            result = strip_img(result, skip)
            x, y = result.shape
        if not to_txt:
            background = np.full((int(out_size * cell_size * scale * y), int(out_size * cell_size * scale * x)), 255, np.uint8)
            for i in range(x):
                for j in range(y):
                    pos = result[i, j]
                    k = pos / len(gray)
                    t = gray[pos - k * len(gray)]
                    background = cv2.putText(background, chars[k], (int(i * out_size * cell_size + out_size * 2) * scale, int(j * out_size * cell_size + out_size * cell_size - out_size * 3) * scale)
                                             , font, out_size * font_size, t, 2)
            background = cv2.resize(background, (0, 0), fx=1.0 / scale, fy=1.0 / scale)
            img_result.append(background)
        else:
            background = [[0 for m in range(x)] for n in range(y)]
            for i in range(x):
                for j in range(y):
                    pos = result[i, j]
                    k = pos / len(gray)
                    t = gray[pos - k * len(gray)]
                    background[j][i] = chars[k]
            img_result.append(background)
    if len(img_result) > 1:
        img_result = cv2.merge(img_result)
    else:
        img_result = img_result[0]
    return img_result


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    # if ustring == ' ':
    #     return '  '
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring


def process_item(args):
    global gray, to_txt, strip, max_dimension
    print locale.getpreferredencoding()
    filename = args[0].decode(locale.getpreferredencoding())
    print filename
    out_filename = None
    max_dimension = float(args[1])
    channel = 3
    if len(args) > 2:
        out_filename = args[2].decode(locale.getpreferredencoding())
        if len(args) > 3:
            channel = int(args[3])
            if len(args) > 4:
                if int(args[4]) == 1:
                    gray = [100, 200]
                    channel = 1
                    to_txt = True
                if len(args) > 5:
                    strip = args[5] == '1'
    global font, scale
    items = []
    for c in chars:
        for v in gray:
            back = np.full((cell_size * scale, cell_size * scale), 255, np.uint8)
            text = cv2.putText(back, c, (2, cell_size * scale - 3 * scale), font, font_size, v, 2)
            text = cv2.resize(text, (0, 0), fx=1.0 / scale, fy=1.0 / scale)
            hog_text = hog(text)
            items.append(hog_text[0, 0])
    print len(items)
    if filename.endswith('gif'):
        gif = VideoFileClip(filename)
        fps = gif.fps
        frames = gif.iter_frames()
        output = []
        num = 1
        for frame in frames:
            print "processing frame " + str(num)
            if channel == 1:
                img = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            else:
                img = frame
            background = process_single_img(img, items)
            output.append(background)
            num += 1
        if out_filename is None:
            out_filename = 'output.gif'
        ImageSequenceClip(output, fps).write_gif("temp__file__.gif")
        shutil.move("temp__file__.gif", out_filename.encode(locale.getpreferredencoding()))
    elif filename.endswith('mp4') or filename.endswith('flv'):
        video = VideoFileClip(filename.encode(locale.getpreferredencoding()))
        fps = video.fps
        audio = video.audio
        frames = video.iter_frames()
        print fps, video.duration
        output = []
        num = 1
        for frame in frames:
            print "processing frame " + str(num)
            if channel == 1:
                img = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            else:
                img = frame
            background = process_single_img(img, items)
            output.append(background)
            num += 1
        if out_filename is None:
            out_filename = 'output.mp4'
        ImageSequenceClip(output, fps).write_videofile("temp__file__.mp4", audio=audio)
        shutil.move("temp__file__.mp4", out_filename.encode(locale.getpreferredencoding()))
    else:
        img = cv2.imread(filename.encode(locale.getpreferredencoding()))
        if channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        print img.shape
        background = process_single_img(img, items)
        if out_filename is None:
            out_filename = 'output.jpg'
        if not to_txt:
            cv2.imwrite(out_filename.encode(locale.getpreferredencoding()), background)
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.imshow("output", background)
            cv2.waitKey(0)
        else:
            with open(out_filename.encode('utf-8'), 'w') as f:
                skip = strB2Q(' ')
                left = ''
                for line in background:
                    for char in line:
                        content = strB2Q(char)
                        if content == skip:
                            left += content
                        else:
                            f.write(left.encode('utf-8'))
                            f.write(content.encode('utf-8'))
                            left = ''
                    left = ''
                    f.write('\n')

# class App:
#     def __init__(self, master):
#         #构造函数里传入一个父组件(master),创建一个Frame组件并显示
#         self.frame = Frame(master)
#         frame = self.frame
#         frame.pack()
#         #创建两个button，并作为frame的一部分
#         self.text = Label(frame, text="文件")
#         self.text.pack(side=LEFT)
#         self.entry = Entry(frame)
#         self.entry.pack(side=LEFT) #此处side为LEFT表示将其放置 到frame剩余空间的最左方
#         self.hi_there = Button(frame, text="选择路径", command=self.show_file)
#         self.hi_there.pack(side=LEFT)

#     def show_file(self):
#         filed = FileDialog(self.frame, u"选择文件")
#         filed.go(pattern=["*.jpg", "*.png", "*.gif"])
#         print "hi there, this is a class example!"


if __name__ == '__main__':
    args = sys.argv[1:]
    process_item(args)
