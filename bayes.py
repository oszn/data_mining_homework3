from sklearn import datasets
iris=datasets.load_iris()
import glob
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
def max_pooling(img):
    width,height=img.shape
    tp=[]
    # print(len(img),len(img[0]))
    for i in range(2,width-2,2):
        for j in range(2,height-2,2):
            p=[]

            for c in range(1,3):
               # print(c)
               p.append(max(img[i-1][j],img[i-1][j-1],img[i-1][j+1]
                        ,img[i][j-1],img[i][j+1]
                        ,img[i+1][j],img[i+1][j-1],img[i+1][j-1]))
            tp.append(max(p))
    return tp

def black_grey(path):
    # print(path)
    img=cv2.imread(path)
    width,height=img.shape[:2][::-1]
    img=cv2.resize(img,(211,141),interpolation=cv2.INTER_CUBIC)
    img_resize=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # cv2.imshow("or",img)
    # cv2.imshow("ig",img_resize)
    # cv2.waitKey()
    # print(img_resize.shape)
    p=[]
    m_pol=max_pooling(img_resize)
    for i in img_resize:
        for j in i:
            p.append(j)
    # print(len(p))
    width,height=img_resize.shape
    return m_pol,width,height
def rg3b():
    pass
lables={"Angry":1,"Disgust":2,"Fear":3,"Happy":4,"Neutral":5,"Sad":6,"Surprise":7}
def lable(path):
    return path.split('\\')[1]
def pac(data,test_data):
    data=np.array(data)
    pca=PCA(n_components=0.99)
    print(data.shape)
    pca.fit(data)
    d=pca.transform(data)
    t_d=pca.transform(test_data)
    # return d
    print(pca.explained_variance_ratio_)
    return d,t_d
def type_path(paths):
    p=[]
    lab=[0,0,0,0,0,0,0,0]
    for i in paths:
        lab[lables[lable(i)]]+=1
        if(lab[lables[lable(i)]]<=600):
            p.append(i)
    test_path=[]
    trian_path=[]
    for i in range(0,len(p),600):
        trian_path.append(p[i:i+400])
        test_path.append(p[i+400:i+600])
    return trian_path,test_path
def load_pca_data():
    f = open("pca.txt", "r")
    lines = f.readlines()
    data = []
    y = []
    for line in lines:
        p = line.split(" ")
        # print(p[-1])
        temp = []
        for x in range(len(p) - 1):
            # print(float(p[x]))
            # print(x)
            temp.append(float(p[x]))
        data.append(temp)
        y.append(int(p[-1].replace("\n", "")))
    train_x = []
    train_y = []
    t_x = []
    t_y = []
    for i in range(0, 2400, 400):
        for i in range(i, i + 200):
            train_x.append(data[i])
            train_y.append(y[i])
        for i in range(i + 200, i + 400):
            t_x.append(data[i])
            t_y.append(y[i])
    return train_x,train_y,t_x,t_y
def bayes(train_x,train_y,t_x,t_y):
    # train_x,train_y,t_x,t_y=load_pca_data()
    clf=GaussianNB()
    clf.fit(train_x,train_y)
    dp=clf.predict(t_x)
    ok=0
    nok=0

    for i in range(len(dp)):
        if dp[i]==t_y[i]:
            ok+=1
        else:
            nok+=1
    print("bayes:"+str(ok/(ok+nok)))
def smv(train_x,train_y,t_x,t_y):
    # train_x, train_y, t_x, t_y = load_pca_data()
    clf=svm.SVC()
    clf.fit(train_x,train_y)
    dp = clf.predict(t_x)
    ok = 0
    nok = 0
    for i in range(len(dp)):
        if dp[i] == t_y[i]:
            ok += 1
        else:
            nok += 1
    print("svm:"+str(ok / (ok + nok)))
# smv()
# bayes()
def load_data(paths):
    x=[]
    y=[]
    p = []
    for i in paths:
        for j in i:
            p.append(j)
    f = 1
    for i in p:
        if (f % 100 == 0):
            print(f)
        f += 1
        d, w, h = black_grey(i)
        x.append(d)
        y.append(lables[lable(i)])
    return x,y
def load_file():
    x=[]
    y=[]
    paths=glob.glob("Train/*/*/*.png")
    m_len=0
    print(len(paths))
    train_paths,test_path=type_path(paths)
    train_x,trian_y=load_data(train_paths)
    test_x,test_y=load_data(test_path)
    train_data,test_data=pac(train_x,test_x)
    smv(train_data,trian_y,test_data,test_y)
    bayes(train_data,trian_y,test_data,test_y)
    # for i in range(len(data)):
    #     for j in data[i]:
    #         f.write(str(j)+" ")
    #     f.write(str(y[i])+"\n")
    # f.close()
#     train_x=[]
#     train_y=[]
#     t_x=[]
#     t_y=[]
#     for i in range(0,2400,400):
#         for  i in range(i,i+200):
#             train_x.append(data[i])
#             train_y.append(y[i])
#         for i in range(i+200,i+400):
#             t_x.append(data[i])
#             t_y.append(y[i])
#     clf.fit(train_x,train_y)
#     dp=clf.predict(t_x)
#     print(dp)
#     print(t_y)
#     print(len(dp),len(t_y))
#     ok=0
#     nok=0
#
#     for i in range(len(dp)):
#         if dp[i]==t_y[i]:
#             ok+=1
#         else:
#             nok+=1
#     print(ok/(ok+nok))
load_file()
# black_grey("Train/Angry/000046280/5.png")