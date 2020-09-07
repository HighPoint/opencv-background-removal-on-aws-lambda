import json
import cv2 as cv
import base64
from os import listdir
import boto3
import numpy as np

def lambda_handler(event, context):

  print(listdir("/opt/"))
  (imageDataString, showDetectObject, showAWSRekognition,
    minConfidence, maxLabels) = loadInitialParameters(event)

  if imageDataString != "":
    image, grayImage = readImageDataString(imageDataString)

    image = processImage(image, grayImage, showDetectObject, showAWSRekognition,
                  maxLabels, minConfidence)

    return returnJSON(image)
  else:
    return False


# processImage

def processImage(image, grayImage, showDetectObject, showAWSRekognition,
                  maxLabels, minConfidence):

  if showDetectObject:
    (dnnModelResponse, masks) = dnnModel(image, maxLabels, minConfidence)
    image = dnnShowMask(dnnModelResponse, masks, image, maxLabels, minConfidence)

  return image

# Show DNN Mask

def dnnShowMask(boxes, masks, image, maxLabels, minConfidence):

  (H, W) = image.shape[:2]

  i = 0
  clone = np.zeros((H, W, 4))

  clone[:,:,0:3] = image.copy()

  for detection in boxes[0,0,:,:]:
    i += 1
    classID = int(detection[1])
    confidence = float(detection[2])

    if confidence > minConfidence:

      box = detection[3:7] * np.array([W, H, W, H])
      (startX, startY, endX, endY) = box.astype("int")
      boxW = endX - startX
      boxH = endY - startY

      mask = masks[i -1, classID]
      mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_LANCZOS4) #cv.INTER_CUBIC

      visMask = getGrabCutMaskPartImage(image, mask, startX, startY, endX, endY)
      visMask = np.where((visMask==2)|(visMask==0),0,255).astype('uint8')

      clone[startY:endY, startX:endX, 3] = np.where((visMask == 255) | (clone[startY:endY, startX:endX, 3]==255), 255, 0).astype('uint8')

  clone = smoothImageEdges(clone)

  return clone


# Smooth Image Edges

def smoothImageEdges(image):

  kernel = kernel = np.ones((9,9),np.uint8)
  closing = cv.morphologyEx(image[:,:,3], cv.MORPH_CLOSE, kernel)

  image[:,:,3] = closing.astype("int")

  blur = cv.GaussianBlur( image[:,:,3],(9,9),0)

  image[:,:,3] = blur.astype("int")

  return image


# Grab Cut for part of the image.

def getGrabCutMaskPartImage(image, mask, startX, startY, endX, endY):

  mask2 = np.zeros(mask.shape[:2], dtype="uint8")
  mask3 = np.zeros(image.shape[:2], dtype="uint8")

  mask2[mask < 0.15] = 0
  mask2[mask >= 0.15] = 2
  mask2[mask >= 0.45] = 3
  mask2[mask >= 0.85] = 1

#  getDistribution(mask2, "mask2 2")

  fgModel = np.zeros((1, 65), dtype="float")
  bgModel = np.zeros((1, 65), dtype="float")

  rect = (startX, endY, endX, startY)

  cloneRegion = image[startY:endY, startX:endX, 0:3].copy()

  rect = (startY,startX,endY,endX)
  (mask2, bgModel, fgModel) = cv.grabCut(cloneRegion, mask2, None, bgModel, fgModel, 5, mode=cv.GC_INIT_WITH_MASK)

# getDistribution(mask2, "mask2")

  return mask2


# Helper Print Function

def getDistribution(image, aString):

  (unique, counts) = np.unique(image, return_counts=True)

  print(f"{aString} unique = {unique} counts = {counts}")

  return


# Run the DNN Model against the imported image (image)
# The DNN Model is stored in the AWS Lambda Layer

def dnnModel(image, maxLabels, minConfidence):

  cvNet = cv.dnn.readNetFromTensorflow('/opt/frozen_inference_graph.pb', '/opt/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

  cvNet.setInput(cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))

  (dnnModelResponse, masks) = cvNet.forward(["detection_out_final", "detection_masks"])

  return (dnnModelResponse, masks)


# The DNN uses the COCO Model labels
def getCOCOModelList():

  f = open("/var/task/Data/COCO_model_label.txt", "r")

  list = []
  for x in f:
    list.append(x)

  f.close();

  return list


# Return the COCO Mode Name from the list

def getCOCOModelName(COCO_model_list, aClass):

  line = COCO_model_list[aClass +1]  # the COCO model list for the Faster_RCNN
  string = str(line).split(" ", 1)[1]

  return string[:-1]

# Resize the Image to the correct size.
# The DNN uses Mask_RCNN. This model is limited to 300 x 300.

def resizeImage(image, maxSize):
  height = image.shape[0]
  width = image.shape[1]

  maxDimension = max(height,width)

  if maxDimension > maxSize:
    scale = maxSize / maxDimension
    width = int(width*scale)
    height = int(height*scale)
    dsize = (width, height)
    image = cv.resize(image, dsize)

  return image

# Convert image to utf-8 encoded base64.
# First write the image

def convertImageToBase64(image):

  cv.imwrite("/tmp/image.png", image)

  with open("/tmp/image.png", "rb") as imageFile:
    str = base64.b64encode(imageFile.read())
    encoded_image = str.decode("utf-8")

  return encoded_image


# Load the inital parameters

def loadInitialParameters(dict):

    imageDataString = dict.get('imageData',"")
    showDetectObject = dict.get('detectObject', True)
    showAWSRekognition = dict.get('awsRekognition', False)
    minConfidence = float(dict.get('confidenceLevel', "70"))/100

    maxLabels = 10

    return imageDataString, showDetectObject, showAWSRekognition, minConfidence, maxLabels


# the return JSON

def returnJSON(image):

  encoded_image = convertImageToBase64(image)

  return {
      "isBase64Encoded": True,
      "statusCode": 200,

      "headers": { "content-type": "image/png",
                   "Access-Control-Allow-Origin" : "*",
                   "Access-Control-Allow-Credentials" : True
      },
      "body":   encoded_image
    }


# readImageDataString

def readImageDataString(imageDataString):

  with open("/tmp/photo.jpg", "wb") as fh:
    fh.write(base64.b64decode(imageDataString))

  # Read the image
  image = cv.imread("/tmp/photo.jpg")
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  return image, gray
