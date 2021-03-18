import os

create=False
if create:
    file_obj = open(r"annotations_.txt","r")
    #file_obj.write("test")
    lines=file_obj.readlines()
    while lines[0][0] != lines[1][0]:
        lines[0]=lines[0][1:]
    if not os.path.isdir('datasets'):
            os.mkdir('datasets')
    listfile=open("datasets/listfile.txt",'w')
    fname_old = None

    for line in lines:
        tokens1=line.split(",")
        tokens2=line.split()
        fname_orig=tokens1[0]
        fname = fname_orig.replace('.png','.txt').replace('.jpg','.txt')
        
        
        file_obj=open("datasets/"+fname,"a")

        if fname_orig!=fname_old or fname_old == None:
            
            listfile.write(','.join([fname_orig,fname]))
            listfile.write('\n')
        
        file_obj.write(','.join(tokens1[1:]))
        
        fname_old=fname_orig
        file_obj.close()
    listfile.close()

import cv2

listfile=open("datasets/listfile.txt",'r')
lines=listfile.readlines()
for line in lines[:10]:
    filenames=line.split(',')
    img=cv.imread('datasets/'+filenames[0])
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()