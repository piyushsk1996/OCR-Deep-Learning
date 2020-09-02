import importlib
import cv2 as cv2
import numpy as np
import mxnet as mx
import random
import matplotlib.pyplot as plt

import matplotlib.patches as patches

from ocr.utils.expand_bounding_box import expand_bounding_box

from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.iam_dataset import resize_image, crop_handwriting_page

from ocr.utils.beam_search import ctcBeamSearch

import ocr.utils.denoiser_utils
import ocr.utils.beam_search

importlib.reload(ocr.utils.denoiser_utils)

importlib.reload(ocr.utils.beam_search)

# Importing SSD and segmentation network
from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()


# Resizing Image to fit the SSD model

def resize_image(image, desired_size):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size

    bounding box: (int, int, int, int)
        (x, y, w, h) in percentages of the resized image of the original
    '''
    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = image[0][0]
    if color < 230:
        color = 230
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
    crop_bb = (left / image.shape[1], top / image.shape[0], (image.shape[1] - right - left) / image.shape[1],
               (image.shape[0] - bottom - top) / image.shape[0])
    image[image > 230] = 255
    return image, crop_bb


MAX_IMAGE_SIZE_FORM = (1120, 800)
MAX_IMAGE_SIZE_LINE = (60, 800)
MAX_IMAGE_SIZE_WORD = (30, 140)


# this function takes in the img file path
def _pre_process_image(img_in, _parse_method):
    im = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
    if np.size(im) == 1:  # skip if the image data is corrupt.
        return None
    # reduce the size of form images so that it can fit in memory.
    if _parse_method in ["form", "form_bb"]:
        im, _ = resize_image(im, MAX_IMAGE_SIZE_FORM)
    if _parse_method == "line":
        im, _ = resize_image(im, MAX_IMAGE_SIZE_LINE)
    if _parse_method == "word":
        im, _ = resize_image(im, MAX_IMAGE_SIZE_WORD)
    img_arr = np.asarray(im)
    return img_arr


######################################################### Plotting Sample Image ###################################################

# Getting Image array
image = _pre_process_image('sampels/2.jpg', 'form')

# Plotting Sample Image
fig = plt.figure(figsize=(4, 4), dpi=500)
axs = plt.gca()

axs.imshow(image, cmap='Greys_r')
axs.axis('off')
print("Plotting Sample Image")
plt.show()

###################################################### Segmenting the paragraph ###########################################


# Initiating Instance of SegmentationNetwork
paragraph_segmentation_net = SegmentationNetwork(ctx=ctx)
# Loading Parameters
paragraph_segmentation_net.cnn.load_parameters("models/paragraph_segmentation2.params", ctx=ctx)
# Calling hybridize method to stabilize image
paragraph_segmentation_net.hybridize()

form_size = (1120, 800)

predicted_bbs = []

fig = plt.figure(figsize=(4, 4), dpi=500)
axs = plt.gca()

# Transforming image for passing to the model
resized_image = paragraph_segmentation_transform(image, form_size)
# passing resized image to the paragraph segmentation net
bb_predicted = paragraph_segmentation_net(resized_image.as_in_context(ctx))

bb_predicted = bb_predicted[0].asnumpy()
bb_predicted = expand_bounding_box(bb_predicted, expand_bb_scale_x=0.03,
                                   expand_bb_scale_y=0.03)
predicted_bbs.append(bb_predicted)

axs.imshow(image, cmap='Greys_r')

(x, y, w, h) = bb_predicted
image_h, image_w = image.shape[-2:]
(x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
rect = patches.Rectangle((x, y), w, h, fill=False, color="r", ls="--")
axs.add_patch(rect)
axs.axis('off')
print("Segmenting Paragraph from image")
plt.show()

################################################### Cropping Segmented Paragraph ##########################################

segmented_paragraph_size = (700, 700)
fig = plt.figure(figsize=(4, 4), dpi=500)
axs = plt.gca()

paragraph_segmented_images = []

# for i, image in enumerate(images):

bb = predicted_bbs[0]
image = crop_handwriting_page(image, bb, image_size=segmented_paragraph_size)
paragraph_segmented_images.append(image)

axs.imshow(image, cmap='Greys_r')
axs.axis('off')
print("Cropping segmented Paragraph")
plt.show()


################################################# Detecting words inside segmented paragraph #####################################

word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
word_segmentation_net.load_parameters("models/word_segmentation2.params")
word_segmentation_net.hybridize()

min_c = 0.1
overlap_thres = 0.1
topk = 600

fig = plt.figure(figsize=(4, 4), dpi=500)
axs = plt.gca()
predicted_words_bbs_array = []

for i, paragraph_segmented_image in enumerate(paragraph_segmented_images):

    predicted_bb = predict_bounding_boxes(
        word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)

    predicted_words_bbs_array.append(predicted_bb)

    axs.imshow(paragraph_segmented_image, cmap='Greys_r')
    for j in range(predicted_bb.shape[0]):
        (x, y, w, h) = predicted_bb[j]
        image_h, image_w = paragraph_segmented_image.shape[-2:]
        (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
        rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
        axs.add_patch(rect)
        axs.axis('off')

plt.show()

###################################################### Detecting Line inside Semented Paragraph ##################################
# line_images_array = []
# fig = plt.figure(figsize=(4, 4), dpi=500)
# axs = plt.gca()
#
# for i, paragraph_segmented_image in enumerate(paragraph_segmented_images):
#
#     axs.imshow(paragraph_segmented_image, cmap='Greys_r')
#     axs.axis('off')
#     axs.set_title("{}".format(i))
#
#     predicted_bbs = predicted_words_bbs_array[i]
#     line_bbs = sort_bbs_line_by_line(predicted_bbs, y_overlap=0.4)
#     line_images = crop_line_images(paragraph_segmented_image, line_bbs)
#     line_images_array.append(line_images)
#
#     for line_bb in line_bbs:
#         (x, y, w, h) = line_bb
#         image_h, image_w = paragraph_segmented_image.shape[-2:]
#         (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
#
#         rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
#         axs.add_patch(rect)
#
# plt.show()
