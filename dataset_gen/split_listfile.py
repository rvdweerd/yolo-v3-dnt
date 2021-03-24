import os


listfile_obj = open("datasets_txt/listfile.txt","r")
lines=listfile_obj.readlines()

splitfile_obj=open("datasets_txt/slistfile.txt","w")
for line in lines:
    img_fname,_ = line.split(",")
    splitfile_obj.write("data/obj/"+img_fname+"\n")

splitfile_obj.close()
listfile_obj.close()
