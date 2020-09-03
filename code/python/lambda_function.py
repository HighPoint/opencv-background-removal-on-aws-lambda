import json
import cv2 as cv
import base64
from os import listdir
import boto3
import numpy as np

def lambda_handler(event, context):

  print(listdir("/opt/"))
  (imageDataString, showDetectObject, showAWSRekognition,
    showLabel, showConfidence,
    minConfidence, maxLabels) = loadInitialParameters(event)

  if imageDataString != "":
    image, grayImage = readImageDataString(imageDataString)

    image = processImage(image, grayImage, showDetectObject, showAWSRekognition,
                  maxLabels, minConfidence, showLabel, showConfidence)

    return returnJSON(image)
  else:
    return False


# processImage

def processImage(image, grayImage, showDetectObject, showAWSRekognition,
                  maxLabels, minConfidence, showLabel, showConfidence):

  if showDetectObject:
    (dnnModelResponse, masks) = dnnModel(image, maxLabels, minConfidence)
    image = dnnShowMask(dnnModelResponse, masks, image, maxLabels, minConfidence)

#    image = dnnShowBoxandLabels(dnnModelResponse, image, maxLabels, minConfidence, showLabel, showConfidence)
#    image = kMeansBGR(image)
#    image = bilateralFilter(image)
#    image = kMeansLAB(image)
#    image = bilateralFilter(image)


#  if showAWSRekognition:
#    f = open("/tmp/photo.jpg","rb")
#    rekognitionLabels = detectObjectRekognition(f.read(), maxLabels, minConfidence)
#    image = rekognizitionShowBoxandLabels(image, rekognitionLabels, (0,0,0), showLabel, showConfidence)


  return image

# Show DNN Mask

def dnnShowMask(boxes, masks, image, maxLabels, minConfidence):

  (H, W) = image.shape[:2]

  i = 0
  clone = np.zeros((H, W, 4))
#  clone[0:int(H/2), 0:int(W/2), 3].fill(255)

  clone[:,:,0:3] = image.copy()
#  clone[:,:, 3].fill(10)

  for detection in boxes[0,0,:,:]:
    i += 1
    classID = int(detection[1])
    confidence = float(detection[2])

    if confidence > minConfidence:
      print(detection)

      box = detection[3:7] * np.array([W, H, W, H])
      (startX, startY, endX, endY) = box.astype("int")
      boxW = endX - startX
      boxH = endY - startY

      mask = masks[i -1, classID]
      mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_NEAREST)
#      mask = (mask > 0.3) #Threshold

      print(f"Mask = {mask.shape}")
      print(f"Image = {image.shape}")

#      roi = image[startY:endY, startX:endX]
#      roi = clone[startY:endY, startX:endX]

#      visMask = (mask * 255).astype("uint8")

      visMask = getEdgeMask(image, mask, startX, startY, endX, endY)
#      instance = cv.bitwise_and(roi, roi, mask=visMask)

#      roi = roi[mask]

#      color = 100
#      blended = ((0.0 * color) + (1.0 * roi)).astype("uint8")

#      clone[startY:endY, startX:endX][mask] = blended
      clone[startY:endY, startX:endX, 3] = visMask

      print(clone.shape)

#      hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#      Hue, Sat, Val = cv.split(hsv)

  return clone


def getEdgeMask(image, mask, startX, startY, endX, endY):

  mask = (mask > 0.3)
  visMask = (mask * 255).astype("uint8")

  cloneRegion = image[startY:endY, startX:endX, 0:3].copy()
  cloneReturn = kMeansBGR(cloneRegion, mask)

  return cloneReturn


def kMeansBGR(image, mask):

  img = image.copy()
#  img = resizeImage(img, 300)

  Z = img.reshape((-1,3))
  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 8
  ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_PP_CENTERS)

  print(f"center = {center}")
  print(f"label shape = {label.shape}")

  # Now convert back into uint8, and make original image


#  center = np.uint8(center)
#  res = center[label.flatten()]
#  res2 = res.reshape((img.shape))

  label[label == 0] = 8
  unique, counts = np.unique(label, return_counts=True)
  print(f"unique = {unique}")
  print(f"counts = {counts}")

  (H, W) = image.shape[:2]

  maskLabel = label.copy()
  mask = (mask > 0.3)

  maskLabel = np.multiply(mask, np.reshape(maskLabel, (H, W)))

  unique2, counts2 = np.unique(maskLabel, return_counts=True)
  print(f"unique2 = {unique2}")
  print(f"counts2 = {counts2}")

  res2 = np.reshape(label, (H, W))

  for i in range(8):
    if(counts2[i+1]/counts[i] > 0.5):
      res2[res2 == (i +1)] = 255
    else:
      res2[res2 == (i+1)] = 0


#  res2 = np.reshape(label, (H, W))
#  res2[res2 == 0] = 8
#  res2[res2 > 3] = 255
#  res2[res2 <= 3] = 0

  print(f"res2 reshape = {res2.shape}")

#  unique, counts = np.unique(label, return_counts=True)
#  print(f"unique = {unique}")
#  print(f"counts = {counts}")

  return res2


def kMeansLAB(image):

  img = image.copy()
  img = resizeImage(img, 300)
  lab_image = cv.cvtColor(img, cv.COLOR_BGR2LAB)

  Z = lab_image.reshape((-1,3))
  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 8
  ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((img.shape))

  return res2



def bilateralFilter(image):

  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#  edges = cv.Canny(gray, 60, 120)
  equalized_gray = cv.equalizeHist(gray)

  gray_filtered = cv.bilateralFilter(equalized_gray, 7, 50, 50)
  edges_filtered = cv.Canny(gray_filtered, 60, 120)

  return edges_filtered


# Run the DNN Model against the imported image (image)
# The DNN Model is stored in the AWS Lambda Layer

def dnnModel(image, maxLabels, minConfidence):

#  (H, W) = image.shape[:2]

  cvNet = cv.dnn.readNetFromTensorflow('/opt/frozen_inference_graph.pb', '/opt/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

  cvNet.setInput(cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))

  (dnnModelResponse, masks) = cvNet.forward(["detection_out_final", "detection_masks"])

  return (dnnModelResponse, masks)



# Show DNN Bounding Box and Optional Labels

def dnnShowBoxandLabels(cvOut, image, maxLabels, minConfidence, showLabel, showConfidence):

  rows = image.shape[0]
  cols = image.shape[1]

  COCO_model_list = getCOCOModelList()

  for detection in cvOut[0,0,:,:]:
    penSize = getImagePenSize(image)
    aConfidence = float(detection[2])

    if aConfidence >= minConfidence:
      aName = getCOCOModelName(COCO_model_list, int(detection[1]))

      left = detection[3] * cols
      top = detection[4] * rows
      right = detection[5] * cols
      bottom = detection[6] * rows
      cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 255), thickness=2*penSize)

      if showLabel or showConfidence:
        writeString = labelString(aName, aConfidence*100, showLabel, showConfidence)
        cv.putText(image, writeString, (int(left), int(top) - 5), cv.FONT_HERSHEY_SIMPLEX,
			      0.5*penSize, (255, 255, 255), 2*penSize)

  return image


# Run AWS Rekognition with the image (imageBytes)

def detectObjectRekognition(imageBytes, maxLabels, minConfidence, region="us-east-1"):

  client=boto3.client('rekognition', region)

  response = client.detect_labels(
    Image= {
        "Bytes": imageBytes
    },
    MaxLabels= maxLabels,
    MinConfidence= minConfidence*100    #AWS Rekognition uses percentage
	)

  labels=response['Labels']

  return labels


# Show the AWS Rekognizition Bounding Box and Optional Labels

def rekognizitionShowBoxandLabels(image, labels, borderColor, showLabel, showConfidence):

  height = image.shape[0]
  width = image.shape[1]
  penSize = getImagePenSize(image)
#  print(f"width = {width} height = {height}")

  for label in labels:
    instances = label.get('Instances')
    for boxes in instances:
      aName = label.get('Name')
      aConfidence = label.get("Confidence")
      aBox = boxes.get('BoundingBox')

      aTop = int(aBox.get('Top') * height)
      aLeft = int(aBox.get('Left') * width)
      aWidth = int(aBox.get('Width') * width)
      aHeight = int(aBox.get('Height') * height)

      cv.rectangle(image, (aLeft, aTop), (aLeft + aWidth, aTop + aHeight), borderColor, 2*penSize)

      if showLabel or showConfidence:
        writeString = labelString(aName, aConfidence, showLabel, showConfidence)
        cv.putText(image, writeString, (aLeft, aTop - 5), cv.FONT_HERSHEY_SIMPLEX,
			      0.5*penSize, borderColor, 2*penSize)

  return image



# Create the Bounding Box Label depending on the showLabel and showConfidence options.

def labelString(labelString, confidenceFloat, showLabel, showConfidence):

  aLabelString = ""

  if showConfidence:
    confidenceString = str(confidenceFloat)[0:4]

  if showLabel and showConfidence:
    aLabelString = labelString + " " + confidenceString
  elif showLabel:
    aLabelString = labelString
  elif showConfidence:
    aLabelString = confidenceString

  return aLabelString


# The DNN uses the COCO Model labels for Faster_RCNN_Inception_v2
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
# The DNN uses Faster_RCNN_Inception_v2. This model is limited to 300 x 300.
# AWS Rekognition is limited to 5MB images.

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
    showLabel = dict.get('showLabel', True)
    showConfidence = dict.get('showConfidence', True)
    minConfidence = float(dict.get('confidenceLevel', "70"))/100

    maxLabels = 10

    return imageDataString, showDetectObject, showAWSRekognition, showLabel, showConfidence, minConfidence, maxLabels


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


# getImagePenSize

def getImagePenSize(image):

  height = image.shape[0]
  width = image.shape[1]

  maxDimension = max(height, width)

  penSize = int(maxDimension/640)
  if penSize == 0:
    penSize = 1

  return penSize


# readImageDataString

def readImageDataString(imageDataString):

  with open("/tmp/photo.jpg", "wb") as fh:
    fh.write(base64.b64decode(imageDataString))

  # Read the image
  image = cv.imread("/tmp/photo.jpg")
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  return image, gray
