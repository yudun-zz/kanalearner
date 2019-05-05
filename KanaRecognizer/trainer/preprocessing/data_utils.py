"""Utilities for loading raw data"""

import struct
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Specify the path to the ETL character database files
ETL_PATH = 'KanaRecognizer/trainer/ETLC_data'


NOISE_PIXEL_NUM_THRESHOLD = 6
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MAIN_DIRECTIONS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def bfs(arr, q, candidates, visited):
    while len(q) > 0:
        point = q.pop(0)
        i = point[0]
        j = point[1]
        if len(candidates) <= NOISE_PIXEL_NUM_THRESHOLD:
            candidates.append((i, j))
        for direction in DIRECTIONS:
            x = i + direction[0]
            y = j + direction[1]
            if 0 <= x < len(arr) and 0 <= y < len(arr[0]) and arr[x][y] == 0 and visited[x][y] == 0:
                visited[x, y] = 1
                q.append((x, y))


def is_noisy(arr, i, j):
    for direction in MAIN_DIRECTIONS:
        x = i + direction[0]
        y = j + direction[1]
        if 0 <= x < len(arr) and 0 <= y < len(arr[0]) and arr[x][y] == 0:
            return False
    return True


# remove noise (dfs points size < threshold) from 64*64 arr
def denoise(arr):
    visited = np.zeros((64, 64))
    for i in range(0, 64):
        for j in range(0, 64):
            if arr[i][j] == 0 and visited[i][j] == 0:
                candidates = []
                q = [(i, j)]
                visited[i, j] = 1
                bfs(arr, q, candidates, visited)
                if len(candidates) <= NOISE_PIXEL_NUM_THRESHOLD:
                    for candidate in candidates:
                        arr[candidate[0]][candidate[1]] = 1

    new_arr = np.ones((64, 64))
    for i in range(0, 64):
        for j in range(0, 64):
            if arr[i][j] == 0 and is_noisy(arr, i, j):
                new_arr[i][j] = 1
            else:
                new_arr[i][j] = arr[i][j]

    return new_arr


def read_record(database, f):
    """Load image from ETL binary
    Args:
        database (string):  'ETL8B2' or 'ETL1C'. Read the ETL documentation to add support
            for other datasets.
        f (opened file): binary file
    Returns:
        img_out (PIL image): image of the Japanese character
    """
    W, H = 64, 63
    if database == 'ETL8B2':
        s = f.read(512)
        r = struct.unpack('>2H4s504s', s)
        i1 = Image.frombytes('1', (W, H), r[3], 'raw')

        # Make the image smaller
        delta_w = 60
        delta_h = 60
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        i1 = ImageOps.expand(i1, padding).resize((64, 63))

        img_out = r + (i1,)
        return img_out

    elif database == 'ETL1C':
        s = f.read(2052)
        r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
        iF = Image.frombytes('F', (W, H), r[18], 'bit', 4)
        iP = iF.convert('RGB')
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(40)

        img_out = r + (iE,)
        return img_out


def get_ETL_data(dataset, categories, writers_per_char,
                 database='ETL8B2',
                 starting_writer=None,
                 vectorize=False,
                 resize=None,
                 img_format=False,
                 ):
    """Load Japanese characters into a list of PIL images or numpy arrays.
    Args:
        dataset (string): the dataset index for the corresponding database. This will be the
            index that shows up in the name for the binary file.
        categories (iterable): the characters to return
        writers_per_char (int): the number of different writers to return, for each character.
        database (str, optional): database name
        starting_writer (int, optional): specify the index for a starting writer
        vectorize (bool, optional): True will return as a flattened numpy array
        resize (tuple, optional): (W,H) tuple to specify the output image dimensions
        img_format (bool, optional): True will return as PIL image
        get_scripts (bool, optional): True will also return a label for the type of Japanese script
    Returns:
        output (X, Y, scriptTypes]): tuple containing the data, labels, and the script type
    """

    W, H = 64, 64
    new_img = Image.new('1', (W, H))

    if database == 'ETL8B2':
        name_base = ETL_PATH + '/ETL8B/ETL8B2C'
    elif database == 'ETL1C':
        name_base = ETL_PATH + '/ETL1/ETL1C_'

    filename = name_base + str(dataset)

    X = []
    Y = []
    scriptTypes = []

    try:
        iter(categories)
    except:
        categories = [categories]

    for id_category in categories:
        with open(filename, 'rb') as f:
            if database == 'ETL8B2':
                f.seek((id_category * 160 + 1) * 512)
            elif database == 'ETL1C':
                f.seek((id_category * 1411) * 2052)

            for i in range(writers_per_char):
                # skip records
                if starting_writer:
                    for j in range(starting_writer):
                        read_record(database, f)

                # start outputting records
                r = read_record(database, f)
                new_img.paste(r[-1], (0, 0))
                iI = Image.eval(new_img, lambda x: not x)

                if database == 'ETL1C':
                    new_arr = denoise(np.uint8(np.array(iI)))
                    iI = Image.new('1', (64, 64))
                    iI.putdata(new_arr.reshape((64 * 64)))
                    if dataset == 9 and id_category == 4 and r[2] == 2672:
                        print(decode_JISX0201(r[3]), 'is missing in sheet 2672')
                        continue

                    if dataset == 12 and id_category == 1 and r[2] == 2708:
                        print(decode_JISX0201(r[3]), 'is missing in sheet 2708')
                        continue

                # resize images
                if resize:
                    # new_img.thumbnail(resize, Image.ANTIALIAS)
                    iI.thumbnail(resize)
                    shapes = resize[0], resize[1]
                else:
                    shapes = W, H

                # output formats
                if img_format:
                    outData = iI
                elif vectorize:
                    outData = np.asarray(iI.getdata()).reshape(
                        shapes[0] * shapes[1])
                else:
                    outData = np.asarray(iI.getdata()).reshape(
                        shapes[0], shapes[1])

                X.append(outData)
                if database == 'ETL8B2':
                    # JIS Kanji Code (JIS X 0208)
                    Y.append(decode_JISX0208(r[1]))
                    if id_category < 75:
                        scriptTypes.append(0)
                    else:
                        scriptTypes.append(2)
                elif database == 'ETL1C':
                    # JIS Katakana X0201
                    Y.append(decode_JISX0201(r[3]))
                    scriptTypes.append(1)

            if database == 'ETL8B2':
                print('finish loading hiragana', id_category+1, ': ', Y[-1])
            elif database == 'ETL1C':
                print('finish loading katakana', (int(dataset) - 7) * 8 + id_category + 1, ': ', Y[-1])

    output = []
    if not img_format:
        X = np.asarray(X, dtype=np.int32)
    output += [X]
    output += [Y]
    output += [scriptTypes]

    return output


# https://stackoverflow.com/questions/43239935/convert-jis-x-208-code-to-utf-8-in-python
def decode_JISX0208(jis_x_0208_code):
    return (b'\033$B' + (jis_x_0208_code).to_bytes(2, byteorder='big')).decode('iso2022_jp')


# https://www.sljfaq.org/afaq/encodings.html#encodings-JIS-X-0201
def decode_JISX0201(jis_x_0201_code):
    return (bytes.fromhex('8e') + struct.pack("B", jis_x_0201_code)).decode('euc_jp')


if __name__ == '__main__':
    # hiragana
    # chars, labs, spts = get_ETL_data(1, range(0, 75), 160, img_format=True)
    # print("hiragana num:", len(chars) / 160)
    # for i in range(0, 75):
    #     idx = i*160 + 4
    #     chars[idx].save("test_hiragana_img/hiragana_"+str(i)+"_"+labs[idx]+".png", 'PNG')

    # katakana
    katakana_num = 0
    characters = []
    labels = []
    scripts = []
    for i in range(7, 14):
        if i < 10:
            filename = '0' + str(i)
        else:
            filename = str(i)

        chars, labs, spts = get_ETL_data(filename, range(0, 8 if i < 13 else 3), 160, database='ETL1C', img_format=True)
        print("finish loading ETL1C_" + filename)
        katakana_num += len(chars)
        characters += chars
        labels = np.concatenate((labels, labs), axis=0)
        scripts = np.concatenate((scripts, spts), axis=0)

    print("katakana num:", katakana_num / 160)
    for i in range(0, 51):
        idx = i*3 + 2
        characters[idx].save("test_katakana_img/katakana_"+str(i)+"_"+labels[idx]+".png", 'PNG')
