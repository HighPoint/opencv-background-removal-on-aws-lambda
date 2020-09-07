# OpenCV Background Removal On AWS Lambda


[![OpenCV Background Removal Launch Stack](readme-images/ImageBackgroundRemovalLaunchStack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=OpenCVBackgroundRemovalStack&templateURL=https://mask-rcnn-source.s3.amazonaws.com/template.yaml)

Serverless removal of images backgrounds with OpenCV, using an AWS Lambda.

# How to Use

1. Click the "Image Background Removal Launch Stack" button:

[![OpenCV Background Removal Launch Stack](readme-images/ImageBackgroundRemovalLaunchStack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=OpenCVBackgroundRemovalStack&templateURL=https://mask-rcnn-source.s3.amazonaws.com/template.yaml)


This will bring you to either the Cloudformation UI or the AWS console if you are not signed in. Sign in, if you are not already. From the Cloudformation UI, click "Next" at the bottom of the screen. Repeat clicking "Next" on the two following pages. You will reach a page with this towards the bottom:

![CloudFormation Shot](/readme-images/CloudFormationShot.png?raw=true)

&nbsp;

Checkmark the three "I acknowledgement" statements and select "Create Stack." This will start building the CloudFormation stack.

&nbsp;

2) Navigate to S3. You should see a bucket named "mask-rcnn-demo-xxxxxxxx", where "xxxxxxxx" is the unique stack identifier from CloudFormation. Open this bucket. You will see an "index.html" file. Open this file. This webpage will appear:

![OpenCV Background Removal Screen Shot](readme-images/BackgroundRemovalShot.png?raw=true)

&nbsp;

If you select the "Object Mask Options", you can control which object classes are used. The "Output Visualization Options" allows you to set a minimum confidence threshold for the OpenCV DNN model. 

![OpenCV Background Removal Menu Screen Shot](readme-images/BackgroundRemovalMenuShot.png?raw=true)

&nbsp;

3) Select a photo. Either select one of the stock photos or upload your own image. Select "Detect," if you uploaded an image or changed the image detection options.

![OpenCV Background Removal Detection Screen Shot](read-me-images/BackgroundRemovalDetectionShot.png?raw=true)

&nbsp;

4) Congratulations! It's that easy.

&nbsp;

# Questions

Any questions or suggestions, just add an "Issues" submission to this repository. Thanks.

&nbsp;

Happy Coding!

