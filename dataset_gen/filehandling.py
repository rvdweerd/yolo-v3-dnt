import os

file_obj = open(r"File_Name","r")
#file_obj.write("test")
lines=file_obj.readlines()

if not os.path.isdir('datasets'):
        os.mkdir('datasets')
listfile=open("datasets/listfile.txt",'w')
fname_old = None

for line in lines:
    tokens1=line.split(",")
    tokens2=line.split()
    fname_orig=tokens1[0]
    fname = fname_orig.replace('.png','.txt').replace('.jpg','.txt')
    
    file_obj=open("datasets/"+fname,"w")
    file_obj.write(','.join(tokens1[1:]))

    if fname_orig!=fname_old or fname_old == None:
        listfile.write(','.join([fname_orig,fname]))
        listfile.write('\n')
    fname_old=fname_orig
    file_obj.close()
listfile.close()