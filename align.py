import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import skimage.io as skio

def ssd(img1, img2):
    dif = img1.ravel() - img2.ravel()
    return np.dot(dif, dif)


def ncc(img1, img2):
    img1 = img1-img1.mean(axis=0)
    img2 = img2-img2.mean(axis=0)

    return np.sum(((img1/np.linalg.norm(img1)) * (img2/np.linalg.norm(img2))))


def border_trim(blue, green, red):

    r_index = []
    c_index = []
    # chop rows
    for r in range(len(blue)):
        black_dff = 255 - sum(blue[r])/len(blue[r])
        white_dif = sum(blue[r])/len(blue[r])
        if black_dff < 10 or white_dif < 10:
            r_index.append(r)
    # chop cols
    for r in range(len(blue.T)):
        black_dff = 255 - sum(blue.T[r])/len(blue.T[r])
        white_dif = sum(blue.T[r])/len(blue.T[r])
        if black_dff < 10 or white_dif < 10:
            c_index.append(r)
    # chop rows
    for r in range(len(green)):
        black_dff = 255 - sum(green[r])/len(green[r])
        white_dif = sum(green[r])/len(green[r])
        if black_dff < 10 or white_dif < 10:
            if r not in r_index:
                r_index.append(r)
    # chop cols
    for r in range(len(green.T)):
        black_dff = 255 - sum(green.T[r])/len(green.T[r])
        white_dif = sum(green.T[r])/len(green.T[r])
        if black_dff < 10 or white_dif < 10:
            if r not in c_index:
                c_index.append(r)

    # chop rows
    for r in range(len(red)):
        black_dff = 255 - sum(red[r])/len(red[r])
        white_dif = sum(red[r])/len(red[r])
        if black_dff < 10 or white_dif < 10:
            if r not in r_index:
                r_index.append(r)
    # chop cols
    for r in range(len(red.T)):
        black_dff = 255 - sum(red.T[r])/len(red.T[r])
        white_dif = sum(red.T[r])/len(red.T[r])
        if black_dff < 10 or white_dif < 10:
            if r not in c_index:
                c_index.append(r)

    blue = np.delete(blue, r_index, axis=0)
    blue = np.delete(blue, c_index, axis=1)
    green = np.delete(green, r_index, axis=0)
    green = np.delete(green, c_index, axis=1)
    red = np.delete(red, r_index, axis=0)
    red = np.delete(red, c_index, axis=1)
    return blue, green, red


def exhaustively_search(img1, img2, window=None, score="ncc", c="r"):
    if window is None:
        window = [-15, 15]
    min_score = -1
    max_score = 99999999999
    displacement = [0, 0]
    d1 = np.linspace(window[0], window[1], abs(window[1]-window[0]), dtype=int)
    d2 = np.linspace(window[0], window[1], abs(window[1]-window[0]), dtype=int)

    for i in d1:
        for j in d2:

            shift_img1 = np.roll(img1, [i, j], axis=(0, 1))
            if score == "ncc":
                chose_score = ncc(shift_img1, img2)

                if chose_score > min_score:
                    min_score = chose_score
                    displacement = [i, j]
            else:
                chose_score = ssd(shift_img1, img2)
                if chose_score < max_score:
                    max_score = chose_score
                    displacement = [i, j]
    a_img1 = np.roll(img1, displacement, axis=(0, 1))
    return a_img1, displacement


def pyramid(img1, img2):
    a_img_final = None
    img1_list, img2_list = [], []
    w, h = img1.shape
    img1_list.append(img1)
    img2_list.append(img2)
    while w > 300 or h > 300:
        w = w/2
        h = h/2
        img1 = resize(img1, (int(w), int(h)))
        img2 = resize(img2, (int(w), int(h)))
        img1_list.append(img1)
        img2_list.append(img2)

    window = [-50, 50]
    displacement = [0, 0]
    while len(img1_list) != 0:
        # get next children images
        img1 = img1_list[-1]
        img2 = img2_list[-1]
        # shift the new image by 2 x child's vector
        a_img1 = np.roll(img1, [2*i for i in displacement], axis=(0, 1))
        # search
        a_img_final, displacement = exhaustively_search(a_img1, img2, window=window)
        window = [int(i/2) for i in window]

        img1_list.pop()
        img2_list.pop()
    # final shift
    a_img_final, displacement = exhaustively_search(a_img_final, img2, window=window)
    return a_img_final, window


def main(score="ncc"):
    for root, dirs, files in os.walk("data", topdown=True):
        test_file = ["00125v.jpg", "service-pnp-prok-01800-01886r.jpg",  "01047u.jpg"]
        test_file = ["01047u.jpg"]
        test_file = files
        for f in test_file:

            img_name = f
            im = Image.open("data/" + img_name)
            im = np.asarray(im)

            # directly chop im
            w, h = im.shape
            # im = im[int(w * 0.01):int(w - w * 0.02), int(h * 0.05):int(h - h * 0.05)]

            height = int(np.floor(im.shape[0] / 3))

            b = im[:height]
            g = im[height: 2 * height]
            r = im[2 * height: 3 * height]

            b, g, r = border_trim(b, g, r)

            b = Image.fromarray(b)
            g = Image.fromarray(g)
            r = Image.fromarray(r)

            b1 = b.filter(ImageFilter.FIND_EDGES)
            g1 = g.filter(ImageFilter.FIND_EDGES)
            r1 = r.filter(ImageFilter.FIND_EDGES)

            b1 = np.array(b1)
            g1 = np.array(g1)
            r1 = np.array(r1)
            # b.save("b1.jpg")
            # g.save("g1.jpg")
            # r.save("r1.jpg")
            # skio.imsave("b1.jpg", b)
            # skio.imsave("g1.jpg", g)
            # skio.imsave("r1.jpg", r)

            # naive search
            #ar, _ = exhaustively_search(r, b, score=score, c="r")
            #ag, _ = exhaustively_search(g, b, score=score, c="g")

            # pyramid search - edge
            _, displacement_r = pyramid(r1, b1)
            print(displacement_r)
            _, displacement_g, = pyramid(g1, b1)
            print(displacement_g)
            #
            ar = np.roll(r, displacement_r, axis=(0, 1))
            ag = np.roll(g, displacement_g, axis=(0, 1))
            im_out = np.dstack([ar, g, b])
            #
            colour_img = Image.fromarray(im_out)
            colour_img.save("output_3_edge/"+img_name)
            print(f)

if __name__ == '__main__':
    #naive_approach()
    main()

