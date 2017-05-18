import urllib
import urllib.request
import cv2
import os
import numpy as np

def store_raw_images(paths, links):
    
    

    pic_num = 1

    for link, path in zip(links, paths):
        if not os.path.exists(path):
            os.makedirs(path)
        image_urls = str(urllib.request.urlopen(link).read())
        
        
        for i in image_urls.split('\\n'):
            try:                
                urllib.request.urlretrieve(i, path+"/"+str(pic_num)+".jpg")
                img = cv2.imread(path+"/"+str(pic_num)+".jpg")             
            
                if img is not None:
                    #resized_image = cv2.resize(img, (100, 100))
                    cv2.imwrite(path+"/"+str(pic_num)+".jpg",img)
                    pic_num += 1

            except Exception as e:
                    print(str(e))  


def createNegImages():
    if not os.path.exists('neg'):
        os.makedirs('neg')
    
    for img in os.listdir('orig'):
        try:
            currImgPath = str(img)
            
            newImg = cv2.imread('orig/'+currImgPath,cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(newImg, (300, 300))
            cv2.imwrite("neg/"+currImgPath, resized_img)

        except Exception as e:
            print(str(e))
    
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



def createNegFile():
    file_type = 'neg'
    for img in os.listdir(file_type):            
        line = file_type+'/'+img+'\n'
        with open('bg.txt','a') as f:
            f.write(line)
          
  
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
    
    
    #store_raw_images(paths, links)
    #createNegImages()
    #removeInvalid(paths)
    #createNegFile()


if __name__ == "__main__":

    main()


#Step 1:
# opencv_createsamples -img arrow100_100.png -bg bg.txt -info info/info.txt -maxxangle 1 maxyangle 1 maxzangle 1 -num 5000
#

#Step 2
# opencv_createsamples -info info/info.txt -num 5000 -w 100 -h 100 -vec positives.vec
# 
#Step 3
# opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 40 -h 40


#kevin@Lenovo-IdeaPad-Y410P ~/Documents/Git/robotics_a1/HarrTraining $ opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 40 -h 40

