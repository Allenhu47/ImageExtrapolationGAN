import cv2
import numpy as np
import os

def main():
    i=1
    while i<=431:
        str1=str(i)
        path1="frames/"+  str1.zfill(4)+'.png'
        path2="mask/"+  str1.zfill(4)+'.png'
        img1=cv2.imread(path1)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        img2=cv2.imread(path2)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        
        print(img1.shape)
        print(mask.shape)
        #cv2.imshow("xiaorun",img2)
        #cv2.waitKey(400)
        new_img = cv2.bitwise_and(img1,img1,mask=mask)
        #img = img1*mask_inv
        print(new_img.shape)
        cv2.imwrite("new/"+str1.zfill(4)+'.png',new_img)
        
        i=i+1
if __name__=='__main__':
    main()

  


