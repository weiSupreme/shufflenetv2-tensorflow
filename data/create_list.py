import os

rootdir='/home/zw/Downloads/tzb1229/train/'
sets=['norm','defect']
txt = open('../tzb0104_train_list.txt','w')
for set_ in sets:
    list_=os.listdir(rootdir+set_)
    for imgn in list_:
        str_=rootdir+set_+'/'+imgn+' '+set_+ '\n'
        txt.write(str_)
txt.close()
