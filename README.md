
[//]: # (Image References)

[ImageNet1]: ./git_images/ImageNet1.jpg "ImageNet part 1"
[ImageNet2]: ./git_images/ImageNet2.jpg "ImageNet part 2"

[ToRemove1]: ./invalid/4.jpg "Inavlid Image"
[ToRemove2]: ./invalid/5.jpg "Inavlid Image"
[ToRemove3]: ./invalid/6.jpg "Inavlid Image"


# NotHotdog-Classifier
Do you watch HBO's Silicon Valley? Because I do and I was inspired by Mr. Jian-Yang to make my own not hotdog classifier

"What would you say if I told you there is a app on the market that tell you if you have a hotdog or not a hotdog. It is very good and I do not want to work on it any more. You can hire someone else." - Jian-Yang , 2017


Here is a demonstration of the final version of the product that we are going to be making
[![IMAGE ALT TEXT](http://img.youtube.com/vi/ACmydtFDTGs/0.jpg)](https://youtu.be/ACmydtFDTGs)
[CLICK ME](https://www.youtube.com/watch?v=ACmydtFDTGs)



# Step 1: Collecting data
The very first step in making a classifier is to collect data. Thus we need to find images of hotdogs and not-hotdogs. The best way do this is if you are a professor at Stanford and you give each of your student an assignment to collect images of food for you, but be warned that they might try and steal the idea from you.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/T0FA_69nXjM/0.jpg)](https://youtu.be/T0FA_69nXjM)
[CLICK ME](https://www.youtube.com/watch?v=T0FA_69nXjM)


So if you're not a professor at Stanford you will need to collect the data yourself :(   Since I'm not a professor at Stanford I had to collect the images myself. To do this I just used [ImageNet](http://www.image-net.org/) to search for my images, since ImageNet is like a database of images. To find the hotdog images I just searched for “hotdog”, “chili dog” and “frankfurter” after downloading all of the images it would me about 1857 hotdog images.  For the not-hotdog images I searched for “food”, “furniture”, “people” and “pets” this give me about 4024 not-hotdog images.

Now to actually download these images I need to get the URLs, to do that I just need to click on the tab download tab click on link called URLs
![ImageNet1]

then copy the URL of the page your on which will use in a script that we'll write to download all of these images.
![ImageNet2]

Next we need to write our scripts to download and save all of these images, here is my [python code](https://github.com/kmather73/NotHotdog-Classifier/blob/master/getData.py) for saving the images. 
The function store_raw_images takes in a list of folders names where you want to save the images to from each of the links.
```
def store_raw_images(folders, links):
    pic_num = 1
    for link, folder in zip(links, folders):
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_urls = str(urllib.request.urlopen(link).read())
        
        
        for i in image_urls.split('\\n'):
            try:                
                urllib.request.urlretrieve(i, folder+"/"+str(pic_num)+".jpg")
                img = cv2.imread(folder+"/"+str(pic_num)+".jpg")                         
                
                # Do preprocessing if you want
                if img is not None:
                    // do more stuff here if you want
                    cv2.imwrite(folder+"/"+str(pic_num)+".jpg",img)
                    pic_num += 1

            except Exception as e:
                    print(str(e))  

```

Next I have my main method to drive the code 

```
def main():
    links = [ 
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537' ]
    
    paths = ['pets', 'furniture', 'people', 'food', 'frankfurter', 'chili-dog', 'hotdog']
    
    
    store_raw_images(paths, links)
```

Now just wait for it to download all of those hotdogs!!!

# Step 2: Cleaning the data
At this point we have collected our data now we just need to clean it up a bit. If you take a look at the data you will probably notice that there are some garbage images that we need to remove, images that look like one of the following

![ToRemove1]
![ToRemove2]
![ToRemove3]

To do this let write some more scripts to do this work for us, first we just need to get a copy of the images that we want to remove and place them in a folder called ‘invalid’.

```
def removeInvalid(dirPaths):
    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            for invalid in os.listdir('invalid'):
                try:
                    current_image_path = str(dirPath)+'/'+str(img)
                    invalid = cv2.imread('invalid/'+str(invalid))
                    question = cv2.imread(current_image_path)
                    if invalid.shape == question.shape and not(np.bitwise_xor(invalid,question).any()):
                        os.remove(current_image_path)
                        break

                except Exception as e:
                    print(str(e))
```
Then just check the folder 'food' to and remove any images that might look like a hotdog. Next I made two folders called "hotdog" and "not-hotdog" and placed the 'food', 'furniture', 'pets', 'people' folders in the "not-hotdog" and the 'frankfurter', 'chili-dog', 'hotdog' folders in the 'hotdog' folder.

# Step 3: Preprocessing and Data Augmentation
Before we can feed our data to train our neural net we first need to do some data normalization and some data augmentation. It turns out that we don't have an equal number of hotdog and not hotdog images which is a problem when training a classifier. To fix this problem we can do some data augment by sampling images from each of the class and applying a random rotation and blur to the image to get more data.

This method can be used to greatly increase the amount of data we have since neural nets need a “large” amount of data to get good results. I used this method to get 15000 images from each class:
```
def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))
    
    
def loadBlurImg(path, imgSize):
    img = cv2.imread(path)
    angle = np.random.randint(0, 360)
    img = rotateImage(img, angle)
    img = cv2.blur(img,(5,5))
    img = cv2.resize(img, imgSize)
    return img

def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []
    
    for path in classPath:
        img = loadBlurImg(path, imgSize)        
        x.append(img)
        y.append(classLable)
        
    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize)
        x.append(img)
        y.append(classLable)
        
    return x, y

def loadData(img_size, classSize):
    hotdogs = glob.glob('./hotdog/**/*.jpg', recursive=True)
    notHotdogs = glob.glob('./not-hotdog/**/*.jpg', recursive=True)
    
    
    imgSize = (img_size, img_size)
    xHotdog, yHotdog = loadImgClass(hotdogs, 0, classSize, imgSize)
    xNotHotdog, yNotHotdog = loadImgClass(notHotdogs, 1, classSize, imgSize)
    print("There are", len(xHotdog), "hotdog images")
    print("There are", len(xNotHotdog), "not hotdog images")
    
    X = np.array(xHotdog + xNotHotdog)
    y = np.array(yHotdog + yNotHotdog)
    
    return X, y
 ```   
To normalize our images we convert them to gray scale and then preform [histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization)

```
def toGray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B 
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    
    images = 0.2989*images[:,:,:,0] + 0.5870*images[:,:,:,1] + 0.1140*images[:,:,:,2]
    return images

def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)
    
    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])
    
    images = images.reshape(images.shape + (1,)) 
    return images

def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)
```

# Step 4: Building The Neural Net
My model is a convolutional neural networks with three convolutional layers followed by two fully connected layers. Its based on the CNN from the steering angle model for building self-driving cars built by [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py). Because if its good enough to drive a car it's good enough to detect a hotdog. 


The model includes ELU layers and dropout to introduce nonlinearity:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 128x128x1 Gray scale image  					| 
| Convolution 8x8     	| 4x4 subsampling 								|
| ELU			      	| 							 					|
| 						|												|
| Convolution 5x5	    | 2x2 subsampling								|
| ELU					|												|
| 						|												|
| Convolution 5x5	    | 2x2 subsampling								|
| Flatten 				| 												|
| Dropout				| .2 dropout probability						|
| ELU					|												|
|						|												|
| Fully connected		| output 512   									|
| Dropout				| .5 dropout probability						|
| ELU					|												|
|						|												|
| Fully connected		| output 2   									|
| Softmax               | output 2                                      |

To actually code this up we will use a library called Keras which is built on top of TensorFlow

```
def kerasModel(inputShape):
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4),border_mode='valid', input_shape=inputShape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
```

# Step 5: Training The Neural Net
To train the network we split our data into a tranining set and a test set
```
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
```
Then is as simple to train
```
inputShape = (128, 128, 1)
model = kerasModel(inputShape)
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, nb_epoch=10, validation_split=0.1)
```

# Step 6: The Results
To test the model on the test set we just do
```
metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
```

We find that we get very good results with 98% accuracy.  This would make Jian-Yang proud.

# Step 7: Profit
Now that we are done we can sell it to Periscope and become very rich
[![IMAGE ALT TEXT](http://img.youtube.com/vi/AJsOA4Zl6Io/0.jpg)](https://youtu.be/AJsOA4Zl6Io)
[CLICK ME](https://www.youtube.com/watch?v=AJsOA4Zl6Io)
