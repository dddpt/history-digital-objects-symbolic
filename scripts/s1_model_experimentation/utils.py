IMAGE_RESIZE_SHAPE = 2* [256]

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# For running inference on the TF-Hub module.
import tensorflow as tf
from tensorflow.errors import ResourceExhaustedError

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

# adds scripts/ and src/ folder: so you can import scripts/functions across project steps
import sys 
sys.path.append("../../src")
sys.path.append("../../scripts")

from os import listdir
from os import path

import pandas as pd
from tqdm.notebook import tqdm

import imagesize

from data_filepaths import image_folders, portraits_csv, images_with_boxes_folder, object_detection_results_csv_path
from utils import *

print(f"utils.py -> os.environ[CUDA_VISIBLE_DEVICES]=|{os.environ['CUDA_VISIBLE_DEVICES']}|")

# experimentation: configuring tensorflow...
#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
#gpus



def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img





def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)






def resize_image(path, new_width=IMAGE_RESIZE_SHAPE[0], new_height=IMAGE_RESIZE_SHAPE[1],
                              display=False, verbose=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  pil_image = Image.open(path)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  if verbose:
        print("Image resized to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


"""
keeping height/width ratio R while keeping pixel count under C
w * h <= C
w/h = R

w=R*h

R*h^2 = C

h = sqrt(C/R)
w = C/h = C / sqrt(C/R)
"""
def ensure_image_size_lower_than_max_pixel_count(path, max_pixel_count=1024**2):
    width, height = imagesize.get(path)
    ratio = width/height
    pixel_count = width * height

    if pixel_count<max_pixel_count:
        return path
    else:
        new_height = int(np.floor(np.sqrt(max_pixel_count/ratio)))
        new_width = int(np.floor(max_pixel_count / new_height))
        return resize_image(path, new_width, new_height)

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

    
    
    

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    #print("Font not found, using default font.")
    font = ImageFont.load_default()

  image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image, image_pil







def format_detection_results(result, name):
    formatted_result =  result.copy()
    for k,v in result.items():
        formatted_result
    # dict_keys(['detection_class_entities', 'detection_scores', 'detection_boxes', 'detection_class_names', 'detection_class_labels'])
    for i, col in enumerate(["x0", "y0", "x1", "y1"]):
        formatted_result[col] = [box[i] for box in formatted_result["detection_boxes"]]
    del formatted_result["detection_boxes"]
    del formatted_result["detection_class_names"]
    dtf = pd.DataFrame(formatted_result)
    dtf["new_filename"] = name
    return dtf






def run_detector(detector, path, name, display=True, max_pixel_count = None, verbose=False):
    original_path = path
    if max_pixel_count is not None:
        path = ensure_image_size_lower_than_max_pixel_count(path, max_pixel_count)
    
    
    img = load_img(path)
    
    
    try:
        converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()
    except ResourceExhaustedError:
        resized = "picture was resized" if original_path!=path else "original image dimension kept"
        print(f"run_detector() ResourceExhaustedError for image {original_path} of dimension {img.shape} ({resized})")
        return None, None

    result = {key:value.numpy() for key,value in result.items()}

    if verbose: 
        print("Found %d objects." % len(result["detection_scores"]))
        print("Inference time: ", end_time-start_time)

    image_with_boxes, image_with_boxes_pil = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])

    if display:
        display_image(image_with_boxes)
    return image_with_boxes_pil, format_detection_results(result, name)


def detect_save_objects_and_image_with_box(detector, image_path, images_with_boxes_folder, object_detection_results_csv_path, **kwargs):
    
    name = image_path.split("/")[-1]
    image_with_boxes_pil, results = run_detector(detector, image_path, name, display=False, **kwargs)
    if image_with_boxes_pil is None and results is None:
        return None
    image_with_boxes_pil.save(path.join(images_with_boxes_folder, name), format="JPEG", quality=90)
    results.to_csv(object_detection_results_csv_path.replace("<IMAGENAME>", name))
    return results




# efficiently overlooks images that have already been object-detected (i.e. have an image in the images_with_boxes_folder)
def object_detection_on_image_paths_list(detector, images_paths,
                                         images_with_boxes_folder, object_detection_results_csv_path,
                                         verbose=True, **kwargs):
    already_done = set(listdir(images_with_boxes_folder))
    not_yet_done_images_paths = [img_path for img_path in images_paths if path.basename(img_path) not in already_done]
    if verbose:
        print(f"object_detection_on_image_paths_list(): {len(already_done)} images already done, {len(not_yet_done_images_paths)} images left to do out of {len(images_paths)}")
    results_per_image = [
        detect_save_objects_and_image_with_box(detector, img_path, images_with_boxes_folder, object_detection_results_csv_path, **kwargs)
        for img_path in tqdm(not_yet_done_images_paths)
    ]
    return results_per_image