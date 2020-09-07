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
#      print(detection)

      box = detection[3:7] * np.array([W, H, W, H])
      (startX, startY, endX, endY) = box.astype("int")
      boxW = endX - startX
      boxH = endY - startY

      mask = masks[i -1, classID]
      mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_LANCZOS4) #cv.INTER_CUBIC

      visMask = getGrabCutMaskPartImage(image, mask, startX, startY, endX, endY)
      visMask = np.where((visMask==2)|(visMask==0),0,255).astype('uint8')

      clone[startY:endY, startX:endX, 3] = np.where((visMask == 255) | (clone[startY:endY, startX:endX, 3]==255), 255, 0).astype('uint8')

#      print(clone.shape)

  clone = smoothImageEdges(clone)

  return clone

def smoothImageEdges(image):

  kernel = kernel = np.ones((9,9),np.uint8)
  closing = cv.morphologyEx(image[:,:,3], cv.MORPH_CLOSE, kernel)

  image[:,:,3] = closing.astype("int")

  blur = cv.GaussianBlur( image[:,:,3],(9,9),0)

  image[:,:,3] = blur.astype("int")

  return image


def getGrabCutMaskWholeImage(image, mask, startX, startY, endX, endY):

  mask2 = np.zeros(mask.shape[:2], dtype="uint8")
  mask3 = np.zeros(image.shape[:2], dtype="uint8")

  mask2[mask < 0.15] = 0
  mask2[mask >= 0.15] = 2
  mask2[mask >= 0.5] = 3
  mask2[mask >= 0.85] = 1

  getDistribution(mask2, "mask2 2")

  mask3[startY:endY, startX:endX] = mask2

  fgModel = np.zeros((1, 65), dtype="float")
  bgModel = np.zeros((1, 65), dtype="float")

  rect = (startX, endY, endX, startY)

  cloneRegion = image[startY:endY, startX:endX, 0:3].copy()

  print(f"mask2 = {mask2.shape}")
  print(f"cloneRegion = {cloneRegion.shape}")

  print(f"startX = {startX}, startY = {startY}, endX = {endX}, endY = {endY}")

  # beginning if grabcut rectangle

  #rect = (startY,startX,endY,endX)
  #(mask3, bgModel, fgModel) = cv.grabCut(image, mask3, rect, bgModel, fgModel, 5, mode=cv.GC_INIT_WITH_RECT)
  #visMask = mask4[startY:endY, startX:endX].copy()

  # end - if grabcut rectangle

#  (mask3, bgModel, fgModel) = cv.grabCut(cloneRegion, mask2, None, bgModel, fgModel, 5, mode=cv.GC_INIT_WITH_MASK)

  rect = (startY,startX,endY,endX)
  (mask3, bgModel, fgModel) = cv.grabCut(image, mask3, rect, bgModel, fgModel, 10, mode=cv.GC_INIT_WITH_MASK)

  mask2 = mask3[startY:endY, startX:endX]

  getDistribution(mask3, "mask3")
  getDistribution(mask2, "mask2")

  return mask2

def getGrabCutMaskPartImage(image, mask, startX, startY, endX, endY):

  getMaskInfo(mask)

  mask2 = np.zeros(mask.shape[:2], dtype="uint8")
  mask3 = np.zeros(image.shape[:2], dtype="uint8")

  mask2[mask < 0.15] = 0
  mask2[mask >= 0.15] = 2
  mask2[mask >= 0.45] = 3
  mask2[mask >= 0.85] = 1

  getDistribution(mask2, "mask2 2")

#  mask3[startY:endY, startX:endX] = mask2

  fgModel = np.zeros((1, 65), dtype="float")
  bgModel = np.zeros((1, 65), dtype="float")

  rect = (startX, endY, endX, startY)

  cloneRegion = image[startY:endY, startX:endX, 0:3].copy()

  print(f"mask2 = {mask2.shape}")
  print(f"cloneRegion = {cloneRegion.shape}")

  print(f"startX = {startX}, startY = {startY}, endX = {endX}, endY = {endY}")

  # beginning if grabcut rectangle

  #rect = (startY,startX,endY,endX)
  #(mask3, bgModel, fgModel) = cv.grabCut(image, mask3, rect, bgModel, fgModel, 5, mode=cv.GC_INIT_WITH_RECT)
  #visMask = mask4[startY:endY, startX:endX].copy()

  # end - if grabcut rectangle

#  (mask3, bgModel, fgModel) = cv.grabCut(cloneRegion, mask2, None, bgModel, fgModel, 5, mode=cv.GC_INIT_WITH_MASK)

  rect = (startY,startX,endY,endX)
  (mask2, bgModel, fgModel) = cv.grabCut(cloneRegion, mask2, None, bgModel, fgModel, 5, mode=cv.GC_INIT_WITH_MASK)

#  mask2 = mask3[startY:endY, startX:endX]

#  getDistribution(mask3, "mask3")
  getDistribution(mask2, "mask2")

  return mask2

def getMaskInfo(image):

  unique = np.unique(image, return_counts=False)

  print(f"mask unique length = {len(unique)} image = {image.shape}")

  return


def getDistribution(image, aString):

  (unique, counts) = np.unique(image, return_counts=True)

  print(f"{aString} unique = {unique} counts = {counts}")

  return


def getEdgeMask(image, mask, startX, startY, endX, endY):

  mask = (mask > 0.5)

  visMask = (mask * 255).astype("uint8")

  cloneRegion = image[startY:endY, startX:endX, 0:3].copy()
  cloneRegion2 = cloneRegion.copy()

  imageRegionCopy = image[startY:endY, startX:endX, 0:3].copy()

#  cloneReturn = kMeansBGR(cloneRegion, mask)

  cloneRegion2 = kMeansGrayscale(cloneRegion2, mask)

  cloneRegion = kMeansHSV(cloneRegion)
  bilateralFilterGrayImage = bilateralFilterGray(cloneRegion, True) #cloneReturn if kMeans

#  equalized_gray = cv.equalizeHist(visMask)
#  gray_filtered = cv.bilateralFilter(equalized_gray, 7, 50, 50)
#  edges_filtered = cv.Canny(visMask, 60, 120)
#  cloneReturn2 = bilateralFilter(visMask)

  cloneRegion.fill(0)

  ret, thresh = cv.threshold(visMask, 127, 255, 0)
  contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  contoursImage = cv.drawContours( cloneRegion, contours, -1, (255,255,255), 3)
  contoursGray = cv.cvtColor(contoursImage, cv.COLOR_BGR2GRAY)

#  contoursGray += cloneReturn2
  contoursGray = getEnhancedEdges(imageRegionCopy, contoursGray, bilateralFilterGrayImage)

  return cloneRegion2 #contoursGray


def getEnhancedEdges(image, contoursGray, cloneReturn2):

  contoursGray += cloneReturn2

  return contoursGray


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

#  print(f"center = {center}")
  print(f"label shape = {label.shape}")

  # Now convert back into uint8, and make original image


  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((img.shape))

#  label[label == 0] = 8
#  unique, counts = np.unique(label, return_counts=True)
#  print(f"unique = {unique}")
#  print(f"counts = {counts}")

#  (H, W) = image.shape[:2]

#  maskLabel = label.copy()
#  mask = (mask > 0.3)

#  maskLabel = np.multiply(mask, np.reshape(maskLabel, (H, W)))

#  unique2, counts2 = np.unique(maskLabel, return_counts=True)
#  print(f"unique2 = {unique2}")
#  print(f"counts2 = {counts2}")

#  res2 = np.reshape(label.copy(), (H, W))
#  mask2 = np.zeros((H, W))

#  for i in range(8):
#    res3 = label.copy()
#    res3 = (res3 == i)

#    res3 = np.reshape(res3, (H, W))
#    res3.astype("uint8")
#    ures = cv.UMat(np.array(res3, dtype=np.uint8))

#    print(f"res3 shape {res3.shape}")
#    output = cv.connectedComponentsWithStats(ures, connectivity=4)

#    num_labels = output[0]
#    labels = output[1]
#    stats = output[2]
#    centroids = output[3]

#    centroid_array = cv.UMat.get(centroids)
#    labels_array = cv.UMat.get(labels)

#    print(f"labels_array shape = {labels_array.shape}")

#    print(f"In {i}, components = {num_labels} ")
#    count = 0
#    for j in range(1, num_labels):
#      centroidXY = centroid_array[j]
#      centroidX = int(centroidXY[0] + 0.5)
#      centroidY = int(centroidXY[1] + 0.5)

#      if mask[centroidY, centroidX] != 0:
#        mask2 += (labels_array == j)
#        count += 1

#    print(f"In num_labels = {num_labels} count = {count}")
 #   unique3, counts3 = np.unique(mask2, return_counts=True)
#    print(f"unique3 = {unique3}")
#    print(f"counts3 = {counts3}")


#  visMask = (mask * 255).astype("uint8")

#  for i in range(8):
#    if(counts2[i+1]/counts[i] > 0.5):
#      res2[res2 == (i +1)] = 255
#    else:
#      res2[res2 == (i+1)] = 0


#  res2 = np.reshape(label, (H, W))
#  res2[res2 == 0] = 8
#  res2[res2 > 3] = 255
#  res2[res2 <= 3] = 0

#  print(f"res2 reshape = {res2.shape}")

#  unique, counts = np.unique(label, return_counts=True)
#  print(f"unique = {unique}")
#  print(f"counts = {counts}")

  return res2


def kMeansHSV(image):

  img = image.copy()
#  img = resizeImage(img, 300)
  lab_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

  Z = lab_image.reshape((-1,3))
  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 8
  ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_PP_CENTERS)
  # Now convert back into uint8, and make original image

  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((image.shape))

  return res2


def kMeansGrayscale(image, mask):

  img = image.copy()
#  img = resizeImage(img, 300)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  equalized_gray = cv.equalizeHist(gray)
  gray_filtered = cv.bilateralFilter(equalized_gray, 7, 50, 50)

  print(f"grayscale_image = {gray_filtered.shape}")

  Z = gray_filtered.reshape((-1,1))

  print(f"Z = {Z.shape}")

  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 8
  ret,label,center=cv.kmeans( Z, K,None,criteria,10,cv.KMEANS_PP_CENTERS)
  # Now convert back into uint8, and make original image

#  print(f"center = {center}")
  print(f"label shape = {label.shape}")

  # Now convert back into uint8, and make original image

  label[label == 0] = 8
  unique, counts = np.unique(label, return_counts=True)
  print(f"unique = {unique}")
  print(f"counts = {counts}")

  (H, W) = image.shape[:2]

  maskLabel = label.copy()
  maskGreater = (mask > 0.5)
  maskLesser = (mask < 0.3)

  maskLabel = np.multiply(maskGreater, np.reshape(maskLabel, (H, W)))

  unique2, counts2 = np.unique(maskLabel, return_counts=True)
  print(f"unique2 = {unique2}")
  print(f"counts2 = {counts2}")

  res2 = np.reshape(label, (H, W))

  for i in range(8):
    if(counts2[i+1]/counts[i] > 0.5):
      res2[res2 == (i +1)] = 1
    else:
      res2[res2 == (i+1)] = 0

  res3 = np.logical_or(res2, maskGreater)
  res3 = (255*res3).astype("uint")

  return res3


def bilateralFilterGray(image, shouldReturnCannyFilter):

  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  equalized_gray = cv.equalizeHist(gray)

  gray_filtered = cv.bilateralFilter(equalized_gray, 7, 50, 50)
  edges_filtered = cv.Canny(gray_filtered, 60, 120)

  if shouldReturnCannyFilter:
    return edges_filtered
  else:
    return gray_filtered


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



# readImageDataString

def readImageDataString(imageDataString):

  with open("/tmp/photo.jpg", "wb") as fh:
    fh.write(base64.b64decode(imageDataString))

  # Read the image
  image = cv.imread("/tmp/photo.jpg")
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  return image, gray
