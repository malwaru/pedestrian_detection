## Essential imports
import glob
import os
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import copy 
import time


class ImageExtractor():
    def __init__(self,
                raw_path='./data/rawdata/leftImg8bit/train/aachen',
                anno_path='./data/annotations/gtBboxCityPersons/train/aachen',
                save_path='./data/test_svm_data/',
                detect_size=(64,128),
                visualise=True,
                save_image=False):
        '''
        Crop out the samples with positive humans
        '''
        self.raw_data_file_path=raw_path
        self.ann_data_file_path=anno_path
        self.save_path=save_path
        self.detect_size=detect_size # Size of the detection for training images
        self.visualise=visualise
        self.sava_image=save_image

        
    def expand_bounds(self,p1,p2):
        '''
        Re-adjust the size of the bounding box to detection size
        
        Todo:
        
        Make sure the boundary expansion does not go outside the image 
        '''
        miss_pixels_x=0
        miss_pixels_y=0
        size_x=abs(p1[0]-p2[0])
        size_y=abs(p1[1]-p2[1])
        # print(f"size x and y {size_x}{size_y}")
        if size_x<self.detect_size[0]:
            miss_pixels_x=self.detect_size[0]-size_x
        elif (size_x%(self.detect_size[0]) != 0):
            miss_pixels_x=size_x-(np.floor(size_x/self.detect_size[0])*self.detect_size[0])
        if size_y<self.detect_size[1]:
            miss_pixels_y=self.detect_size[1]-size_y
        elif ((size_y)%(self.detect_size[1]) != 0):
            miss_pixels_y=size_y-(np.floor(size_y/self.detect_size[1])*self.detect_size[1])
        # print(f"miss x and y {miss_pixels_x}{miss_pixels_y}")
            
        #After accounting for the fact if missing no of pixels is odd  
        #Expand the boundaries
        p1[0]=int(p1[0]-np.floor(miss_pixels_x/2))
        p2[0]=int(p2[0]+(miss_pixels_x-np.floor(miss_pixels_x/2)))
        
        p1[1]=int(p1[1]-np.floor(miss_pixels_y/2))
        p2[1]=int(p2[1]+(miss_pixels_y-np.floor(miss_pixels_y/2)))
        
        return  p1,p2

   
    def extract_positive(self):
        '''
        Extract the positives 
        '''

        iter=1
        # iteration_limit=2
        

        for filename in glob.glob(os.path.join(self.ann_data_file_path,"*.json")):
            ##Break the loop afte iteations for testing    
            # if iter>iteration_limit:
            #     break        
            iter+=1
            
            try:
                file_json=open(filename)
                annotation_data=json.load(file_json)
                bBoxs=[]
                for objs in annotation_data['objects']:
                    if  objs['label']=='pedestrian':
                        bBoxs.append(objs['bbox'])
                # Close the json file 
                file_json.close()

                ##Get base file namee replace with raw image data
                filename=os.path.basename(filename)
                filename=filename.replace('json','png')
                filename=filename.replace('gtBboxCityPersons','leftImg8bit')
                #Find the right file and show original an with boxes
                image_path=os.path.join(self.raw_data_file_path,filename)    
                current_image = cv2.imread(image_path)
                # current_image=cv2.cvtColor(current_image,cv2.COLOR_BGR2RGB)        
                # cv2.imshow("full",current_image)
                current_people_list=[]# Save the current list of bounding boxes in a image

                for i,bBox in enumerate(bBoxs):
                    p1=[bBox[0],bBox[1]]
                    p2=[bBox[0]+bBox[2],bBox[1]+bBox[3]]
                    new_p1,new_p2=self.expand_bounds(p1,p2)
                    img_crop=current_image[new_p1[1]:new_p2[1],new_p1[0]:new_p2[0]]
                    img_crop=cv2.resize(img_crop,(self.detect_size[0],self.detect_size[1]))
                    if self.sava_image:
                        filename=os.path.basename(filename)
                        filename=filename.replace('.png',str(i)+'.png')
                        save_path=self.save_path+'positive/'+filename
                        # cv2.imwrite(save_path,img_crop)
                    current_people_list.append([new_p1,new_p2])
                    if self.visualise:
                        cv2.imshow("positive",img_crop)
                        cv2.waitKey(0)            
                self.extract_negative(current_image,current_people_list,filename)
                

            


            except:
                print(f"Error at image {iter}")

           


        print(f"Iteration is over")
        cv2.destroyAllWindows()



    def extract_negative(self,image,people_list,filename):
            '''
            Extract negatives of the same image
            Take a random point in the image 
            Check if is in the bounds of people list 
            if not add it to a negative image 
            '''
            negative_list=[]
            # for _ in range(2):
            for i,people in enumerate(people_list):
                x_size=abs(people[0][0]-people[1][0])
                y_size=abs(people[0][1]-people[1][1])  
                in_bound=False
                p_x=0
                p_y=0
                iterator=0
                while not in_bound:  
                    #Stop looking for negatives if the iterator is above 500
                    if iterator>500:
                        break
                    p_x=np.random.randint(0,image.shape[0]-self.detect_size[0])
                    p_y=np.random.randint(0,image.shape[1]-self.detect_size[1])
                    p_xs=range(p_x,p_x+x_size)
                    p_ys=range(p_y,p_y+y_size)     
                    bounds=np.zeros(shape=(1,len(people_list))) 
                    for i,check_list in enumerate(people_list):                
                        check_list_x=set(range(check_list[0][0],check_list[1][0]))
                        check_list_y=set(range(check_list[0][1],check_list[1][1]))
                        if (len(check_list_x.intersection(p_xs)) > 0) and len(check_list_y.intersection(p_ys)) > 0:
                            bounds[0,i-1]=1
                    if np.sum(bounds)>0:
                        in_bound=True
                    iterator+=1
                
                # print(f" image {image.shape} px: {p_x} xsize {x_size} py {p_y} ysize {y_size}")
                img_crop=image[p_x:p_x+x_size,p_y:p_y+y_size]
                # print(f" iori image {image.shape} cropmage {img_crop.shape} px: {p_x} xsize {x_size} py {p_y} ysize {y_size}")
                img_crop=cv2.resize(img_crop,(self.detect_size[0],self.detect_size[1]))
                if self.sava_image:                
                    filename=filename.replace('.png',str(i)+'.png')
                    save_path=self.save_path+'negative/'+filename
                    cv2.imwrite(save_path,img_crop)
                if self.visualise:
                    cv2.imshow("negative_crop",img_crop)
                    cv2.waitKey(0)
                negative_list.append([[p_x,p_y],[p_x+x_size,p_y+y_size]])





class PerformanceMetrics():
    def __init__(self) -> None:
        anno_path='./data/annotations/gtBboxCityPersons/train/aachen'

    



# if __name__=='__main__':

#     print(f"Statt")
#     positives_extractor=ImageExtractor()
#     positives_extractor.extract_positive()


