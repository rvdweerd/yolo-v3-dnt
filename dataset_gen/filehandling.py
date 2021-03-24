import os

create_annotations=True
create_video=False

annotation_file="dataset_gen/annotations_pos_br.txt"

if create_annotations:
    file_obj = open(annotation_file,"r")
    #file_obj.write("test")
    lines=file_obj.readlines()
    while lines[0][0] != lines[1][0]:
        lines[0]=lines[0][1:]
    if not os.path.isdir('datasets_txt'):
            os.mkdir('datasets_txt')
    listfile=open("datasets_txt/listfile.txt",'a')
    fname_old = None

    for line in lines:
        tokens1=line.split(",")
        tokens2=line.split()
        fname_orig=tokens1[0].replace('\n',"")#####
        fname = fname_orig.replace('.png','.txt').replace('.jpg','.txt').replace('\n',"")
        
        
        file_obj=open("datasets_txt/"+fname,"a")

        if fname_orig!=fname_old or fname_old == None:
            
            listfile.write(','.join([fname_orig,fname]))
            listfile.write('\n')
        
        str=' '.join(tokens1[1:])
        if str != '':
            file_obj.write(str)
        
        fname_old=fname_orig
        file_obj.close()
    listfile.close()

import cv2
import numpy as np

if create_video:
    listfile=open("datasets_txt/listfile.txt",'r')
    lines=listfile.readlines()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filenames=lines[0].split(',')
    img=cv2.imread('../robodetect-imgs/'+filenames[0])
    h,w,c=img.shape 
    print('img shape',w,h,c)
    video=cv2.VideoWriter('video.avi', fourcc, 15,(640,480))
    #video=cv2.VideoWriter('video.avi', fourcc, 15,(w,h))

    for line in lines:#[:200]:
        filenames=line.split(',')
        img=cv2.imread('../robodetect-imgs/'+filenames[0])
        
        boxfile=open("datasets_txt/"+filenames[1].replace('\n',''))
        lines=boxfile.readlines()
        for line in lines:
            line=line.split(',')
            cat=float(line[0])
            xrel=float(line[1])
            yrel=float(line[2])
            wrel=float(line[3])
            hrel=float(line[4])
            h,w,c=img.shape
            #print(img.shape)
            x0=int((xrel-0.5*wrel)*w)
            y0=int((yrel-0.5*hrel)*h)
            x1=int((xrel+0.5*wrel)*w)
            y1=int((yrel+0.5*hrel)*h)

            cv2.rectangle(img,(x0,y0),(x1,y1) ,(0,255,0),2)
        img=cv2.copyMakeBorder(img,0,480-h,0,640-w,cv2.BORDER_CONSTANT)
        video.write(img)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    video.release()
        