
[//]: # (Image References)

[ImageNet1]: ./git_images/ImageNet1.jpg "ImageNet part 1"
[ImageNet2]: ./git_images/ImageNet2.jpg "ImageNet part 2"

[ToRemove1]: ./invalid/4.jpg "Inavlid Image"
[ToRemove2]: ./invalid/5.jpg "Inavlid Image"
[ToRemove3]: ./invalid/6.jpg "Inavlid Image"


# NotHotdog-Classifier
Do you watch HBO's silicon valley? Because I do and I was inspired by Mr. Jian-Yang to make my own not hotdog classifier

"What would you say if I told you there is a app on the market that tell you if you have a hotdog or not a hotdog. It is very good and I do not want to work on it any more. You can hire someone else." - Jian-Yang , 2017


Here is a demonstration of the final version of the product that we are going to be making
[![IMAGE ALT TEXT](http://img.youtube.com/vi/ACmydtFDTGs/0.jpg)](https://youtu.be/ACmydtFDTGs)
[CLICK ME](https://www.youtube.com/watch?v=ACmydtFDTGs)



# Step 1: Collecting data
The very first step in making a classifier is to collect data. Thus we need to find images of hotdogs and not-hotdogs. The best way do this is if you a professor at Stanford and you give each of your student an assignment to collect images of food for you, but be warned that they might try and steal your Idea.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/T0FA_69nXjM/0.jpg)](https://youtu.be/T0FA_69nXjM)
[CLICK ME](https://www.youtube.com/watch?v=T0FA_69nXjM)


So if you're not a professor at Stanford you will need to collect the data yourself :(   Since Im not a processor at Stanford I had to collect the images myself. To do this I just used [ImageNet](http://www.image-net.org/) to search for my images, since ImageNet is like a database for images. To find the hotdog images I searched for “hotdog”, “chili dog” and “frankfurter” this give me about 1857 images.  For the not-hotdog images I searched for “food”, “furniture”, “people” and “pets” this give about 4024 not-hotdog images.

Now theses are the images that I want to download I just need to click on the tab download tab click on link
![ImageNet1]

then copy the url to get the urls for all of the images
![ImageNet2]

Next we need to write our scripts to download and save all of these images, here is my [python code](https://github.com/kmather73/NotHotdog-Classifier/blob/master/getData.py) for saving the images 
Here the function store_raw_images takes in a list of folders names where you want to save the images to from each of the links.
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

Now just wait for it to download all of thoses hotdogs!!!

# Step 2: Cleaning the data
At this point we have collected our data now we just need to clean it up a bit. If you take a look at the data you will probably notice that there are some garbage images that we need to remove, like images that look like one of the following

![ToRemove1]
![ToRemove2]
![ToRemove3]

To do this let write some more scripts to do this work for use, first we just need to get a copy of the images that we want to remove and place them in a folder called ‘invalid’.

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
Then just check the folder 'food' to and remove any images that might look like a hotdog. Next I made two folders called "hotdog" and "not-hotdog" and placed the 'frankfurter', 'chili-dog', 'hotdog' folders in the "not-hotdog" and the 'frankfurter', 'chili-dog', 'hotdog' folders in the 'hotdog' folder.

# Step 3: Preprocessing and Data Augmentation
Before we can feed out data to train our neural net we first need to do some data normalization and some data augmentation. It turns out that we don't have an equal number of hotdog and not hotdog images which is a problem when training a classifier. To fix this problem we can do some data augment by sampling images from each of the class and applying a random rotation and blur to the image to get more data.

This method can be used to greatly increase the amount of data we have since neural nets need a “large” amount of data to get good results. I used this method to get 10000 images from each class
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

# Step 4: Building our Neural Net
Coming soon...

# Step N: Profit
Now that we are done we can sell it to Periscope and become very rich
[![IMAGE ALT TEXT](http://img.youtube.com/vi/AJsOA4Zl6Io/0.jpg)](https://youtu.be/AJsOA4Zl6Io)
[CLICK ME](https://www.youtube.com/watch?v=AJsOA4Zl6Io)
