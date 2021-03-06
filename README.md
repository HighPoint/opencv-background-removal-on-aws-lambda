# OpenCV Background Removal On AWS Lambda  


[![OpenCV Background Removal Launch Stack](readme-images/ImageBackgroundRemovalLaunchStack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=OpenCVBackgroundRemovalStack&templateURL=https://mask-rcnn-source.s3.amazonaws.com/template.yaml)

Serverless removal of images backgrounds with OpenCV, using an AWS Lambda. 

Sample Human Image Input:

![One women](/readme-images/FaceInput.jpg?raw=true)

Sample Human Image Output:

![One women with background removed](/readme-images/FaceOutput.png?raw=true)

OpenCV Background Removal on AWS Lambda uses a three step method to remove the background. 

First, the python lambda function uses OpenCV's deep neural network (DNN) to identify areas of interest in the image. These areas are given as probability of being part of an object, a person or a dog for example. This results in a highly pixelated image if viewed.

Second, the area probabilities are inputed into the OpenCV GrabCut algorithm. GrabCut looks for edges to make more realistic cuts between the object and the background.

Finally, the image is smoothed using a Gaussian Blur.

Sample Dog Image Input:

![Two dogs](/readme-images/DogInput.jpg?raw=true)

Sample Dog Image Output:

![Dogs with background removed](/readme-images/DogOutput.png?raw=true)

# How to Use

1. Click the "Image Background Removal Launch Stack" button:

[![OpenCV Background Removal Launch Stack](readme-images/ImageBackgroundRemovalLaunchStack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=OpenCVBackgroundRemovalStack&templateURL=https://mask-rcnn-source.s3.amazonaws.com/template.yaml)


This will bring you to either the Cloudformation UI or the AWS console if you are not signed in. Sign in, if you are not already. From the Cloudformation UI, click "Next" at the bottom of the screen. Repeat clicking "Next" on the two following pages. You will reach a page with this towards the bottom:

![CloudFormation Shot](/readme-images/CloudFormationShot.png?raw=true)

&nbsp;

Checkmark the three "I acknowledgement" statements and select "Create Stack." This will start building the CloudFormation stack. It may take several minutes for CloudFormation to finish.

&nbsp;

2) Navigate to S3. You should see a bucket named "mask-rcnn-demo-xxxxxxxx", where "xxxxxxxx" is the unique stack identifier from CloudFormation. Open this bucket. You will see an "index.html" file. Open this file. This webpage will appear:

![OpenCV Background Removal Screen Shot](readme-images/BackgroundRemovalShot.png?raw=true)

&nbsp;

If you select the "Object Mask Options", you can control which object classes are used. The "Output Visualization Options" allows you to set a minimum confidence threshold for the OpenCV DNN model. 

![OpenCV Background Removal Menu Screen Shot](readme-images/BackgroundRemovalMenuShot.png?raw=true)

&nbsp; 

3) Select a photo. Either select one of the stock photos or upload your own image. Select "Detect," if you uploaded an image or changed the image detection options.

![OpenCV Background Removal Detection Screen Shot](readme-images/BackgroundRemovalWorkingShot.png?raw=true)

&nbsp;

4) Congratulations! It's that easy.

&nbsp;

# Output File

OpenCV Background Removal on AWS Lambda returns a 32 bit png file. The png alpha channel is marked either transparent as 0 or visible as 255. The original image is stored in the RGB channels of the png file. This means you can visually restore sections of image by editing the alpha channel.

&nbsp;

# Questions

Any questions or suggestions, just add an "Issues" submission to this repository. Thanks.

&nbsp;

Happy Coding!

