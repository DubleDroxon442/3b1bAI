import tkinter as tk
from tkinter import *
from PIL import Image,ImageDraw
from pynput.mouse import Listener
import numpy as np
from random import *
import scipy
import imageio as imageio


import math
import tensorflow as tf# Imports tensorflow
import keras # Imports keras
from keras.datasets import mnist
import random as random
seed(1)
(train_X, train_y), (test_X, test_y) = mnist.load_data()


###########################################################################
###########################################################################
#||||||||||||||||||AI|||||||||||||||||||||
###########################################################################
###########################################################################


def convert0(temp1):
    temp = []
    for i in range(len(temp1)):
        x1 = ""
        for k in range(len(temp1[i])):
            if (temp1[i][k:k + 1] != "\n"):
                x1 += temp1[i][k:k + 1]
        temp.append(x1)
    return temp
def convert2(y2):
    res = []
    s = ""
    for k in range(len(y2)):
        if (y2[k:k + 1] == ","):
            res.append(float(s))
            s = ""
        elif (y2[k:k + 1] == ";"):
            res.append(float(s))
            break
        else:
            s += y2[k:k + 1]
    return res
def convert3(y1):
    b1 = True
    b2 = False
    i = 0
    res = []
    while (b1):
        res1 = []
        if (y1[i] == "E"):
            b1 = False
        if (y1[i] == "S"):
            b2 = True
            i += 1
        while (b2):
            if (y1[i][0:1] == "F"):
                b2 = False
                res1 = np.array(res1)
                res.append(res1)
            else:
                res1.append(convert2(y1[i]))
            i += 1

    return res


# train_X is the training images
#train_y is the corresponding number to those training images

learning_rate=1.0
class NeuralNetwork:
    learn=None
    weights=None
    biases=None
    numlayers=None
    sizes=None
    activations=None
    z=[]
    error=[]
    desired=[]
    changeWeight=[]
    changeBias=[]
    momWeight=[]
    momBias=[]
    num=0
    s1=[]
    def __init__(self,sizes):
        self.numlayers=len(sizes)
        self.sizes=sizes
        self.biases=[]
        self.weights=[]
        self.z=[]
        self.error=[]
        self.acc=[]
        for i in range(self.numlayers-1):
            self.changeWeight.append(np.zeros((sizes[i+1],sizes[i])))
            self.changeBias.append(np.zeros((sizes[i+1],1)))
            self.momWeight.append(np.zeros((sizes[i + 1], sizes[i])))
            self.momBias.append(np.zeros((sizes[i + 1], 1)))
        for i in range(self.numlayers - 1):
            temp = np.ones((self.sizes[i + 1], self.sizes[i]))
            for r in range(temp.shape[0]):
                for c in range(temp.shape[1]):
                    min = -1.0 / ((2.0 * self.sizes[i]) ** 0.5)
                    max = 1.0 / ((2.0 * self.sizes[i]) ** 0.5)
                    num = min + (random.random() * (max - min))
                    step1 = temp[r][c] * num
                    temp[r][c] = step1
            self.weights.append(temp)
            temp1 = np.ones((self.sizes[i + 1], 1)) * 0.1
            self.biases.append(temp1)

        self.activations=[]
    def printWeights(self):
        print(self.weights)
    def printBiases(self):
        print(self.biases)

    def updateFiles(self):
        file1=open("weights.txt","r+")
        file2=open("biases.txt","r+")
        file1.truncate(0)
        file2.truncate(0)
        s1=""
        s2=""
        for i in range(len(self.weights)):
            s1+="S\n"
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    if (k == self.weights[i].shape[1] - 1):
                        s1 += str(self.weights[i][j, k])+";"
                    else:
                        s1 += str(self.weights[i][j, k]) + ","
                s1+="\n"
            s1+="F\n"
        file1.write(s1)
        file1.write("E")
        for i in range(len(self.biases)):
            s2+="S\n"
            for j in range(self.biases[i].shape[0]):
                for k in range(self.biases[i].shape[1]):
                    if (k == self.biases[i].shape[1] - 1):
                        s2 += str(self.biases[i][j, k])+";"
                    else:
                        s2 += str(self.biases[i][j, k]) + ","
                s2+="\n"
            s2+="F\n"
        file2.write(s2)
        file2.write("E")
        file1.close()
        file2.close()

    def applySave(self):
        file1 = open(r"weights.txt", "r+")
        rl = file1.readlines()
        temp1 = convert3(convert0(rl))
        self.weights=temp1

        file2 = open(r"biases.txt", "r+")
        rl1 = file2.readlines()
        temp2 = convert3(convert0(rl1))
        self.biases = temp2

        file1.close()
        file2.close()

    def output(self,x):
        self.activations = []
        self.z = []
        self.error = []
        self.activations.append(x)
        for i in range(self.numlayers-1):
            temp = np.dot(self.weights[i],self.activations[i]) + self.biases[i]
            self.z.append(temp)
            temp1 = sigmoid_v(temp)
            self.activations.append(temp1)
        return self.activations[len(self.activations)-1]

    def feedforward(self,x,y):
        self.activations=[]
        self.z=[]
        self.error=[]
        self.activations.append(x)
        for i in range(self.numlayers-1):
            temp=np.dot(self.weights[i],self.activations[i])+self.biases[i]
            self.z.append(temp)
            temp1=sigmoid_v(temp)
            self.activations.append(temp1)
        self.errors(y)

    def errors(self,y):
        e0=(self.activations[len(self.activations)-1]-y)*sigp_v(self.z[len(self.z)-1])
        self.error.append(e0)
        self.changeWeight[len(self.changeWeight)-1]+=np.dot(self.error[0],self.activations[len(self.activations)-2].transpose())
        self.changeBias[len(self.changeBias)-1]+=self.error[0]
        for i in range(len(self.weights)-1,0,-1):
            e1=np.dot(self.weights[i].transpose(),self.error[0])*sigp_v(self.z[i-1])
            self.error.insert(0,e1)
            self.changeWeight[i-1]+=np.dot(self.error[0],self.activations[i-1].transpose())
            self.changeBias[i-1]+=self.error[0]


    def SGDMLP(self,data,test,mb,epochs,lr,sched):
        self.acc=[]
        self.learn=lr
        n=0.0
        for e in range(epochs):
            n+=1.0
            random.shuffle(data)
            minibatches = []

            for m in range(int(len(data) / mb)):
                temp = []
                for i in range(mb):
                    temp.append(data[m * mb + i])
                minibatches.append(temp)

            for m in range(len(minibatches)):
                for i in range(len(minibatches[m])):
                    self.feedforward(minibatches[m][i][0], minibatches[m][i][1])
                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] - self.changeWeight[i] * (self.learn / mb)
                    self.biases[i] = self.biases[i] - self.changeBias[i] * (self.learn / mb)
                    self.momWeight[i] = (self.learn / mb)*self.changeWeight[i]
                    self.changeWeight[i] = np.zeros(self.weights[i].shape)
                    self.changeBias[i] = np.zeros(self.biases[i].shape)
            nn.updateFiles()
            r1 = 0.0
            r2 = 0.0
            cost = 0.0
            for t in range(len(test)):
                temp1 = self.output(test[t][0])
                temp = (self.output(test[t][0]) - test[t][1]) * (self.output(test[t][0]) - test[t][1])
                for r in range(temp.shape[0]):
                    cost += temp[r][0]
                ind = 0
                ind1 = 0
                max = temp1[0][0]
                max1 = test[t][0][0]

                for r in range(1, temp1.shape[0]):
                    if (max < temp1[r][0]):
                        max = temp1[r][0]
                        ind = r
                    if (max1 < test[t][1][r][0]):
                        max1 = test[t][1][r][0]
                        ind1 = r
                if (ind == ind1):
                    r1 += 1.0
                r2 += 1.0
            cost /= len(test)
            print("COST: " + str(cost) + " ACCURACY: " + str(r1 / r2))
            self.acc.append(r1/r2)
            if(sched):
                if(n==30.0):
                    self.learn = lr * math.pow(0.5, math.floor((n + 1.0) / 30.0))
                    n=0.0
            stopbool=True
            if(e>5):
                for i in range(4):
                    if(self.acc[len(self.acc)-1]!=self.acc[len(self.acc)-1-(i+1)]):
                        stopbool=False
                        break
                if(stopbool):

                    return self.acc[len(self.acc)-1]



    def printS1(self):
        print(self.s1)
    def getActivations(self):
        return self.activations
    def printChangeBias(self):
        print(self.changeBias)

    def printChangeWeight(self):
        print(self.changeWeight)

    def printError(self):
        print(self.error)

    def printDesired(self):
        print(self.desired)


    def printActivations(self):
        print(self.activations[len(self.activations)-1])

    def printZ(self):
        print(self.z)



###################################################################################################################
###################################################################################################################
#||||||||||||||||||||||Misc Functions||||||||||||||||||||||
###################################################################################################################
###################################################################################################################
def convert(t):
    res=[]
    for r in range(len(t)):
        for c in range(len(t[r])):
            res.append(float(t[r][c])/255.0)
    return res
def converty(i1):
    res=[]
    for i3 in range(10):

        if(i3 ==i1):
            res.append(1.0)
        else:
            res.append(0.0)
    return res
def sigmoid(x):
    if (x == np.nan):
        print("nan")
    if (x == np.inf):
        print(x)
        print("inf")

    return max(-0.1 * x, x)
def sigPrime(x):
    if (np.isinf(x)):
        print(x)
        print("pinf")
    if (x > 0):
        return 1.0
    else:
        return -0.1
sigmoid_v = np.vectorize(sigmoid)
sigp_v=np.vectorize(sigPrime)

###################################################################################################################
###################################################################################################################


train=[]
val=[]
for i in range(len(train_X)):
    temp1=train_X[i].reshape((784,1))/255.0
    temp=np.zeros((10,1))
    temp[train_y[i]][0]=1.0
    train.append((temp1,temp))

for i in range(len(test_X)):
    temp1 = test_X[i].reshape((784, 1)) / 255.0
    temp = np.zeros((10, 1))
    temp[test_y[i]][0] = 1.0
    val.append((temp1, temp))


#nn.SGDMLP(train,val,1000,100,0.1,True)

sizes=[784,400,400,10]
nn = NeuralNetwork(sizes)
nn.applySave()
###################################################################################################################
###################################################################################################################
#||||||||||||||||||||||User Input/GUI||||||||||||||||||||||
###################################################################################################################
###################################################################################################################
w1=300
h1=300
root = tk.Tk()
root.geometry(""+str(w1+180)+"x"+str(h1+50))
pixel=tk.PhotoImage(width=h1,height=w1)
fr=tk.Frame(root,height=h1+50,width=w1+180, bg="white")
cv=tk.Canvas(fr,height=h1,width=w1, bg="black")

img=Image.new("RGB",(w1,h1),color="black")
dimg=ImageDraw.Draw(img)

fr.place(x=0,y=0)
cv.place(x=0,y=0)

#fr.pack()
#cb.pack()
#cv.pack()
draw=False
brush_size=11

def clear():
    dimg.rectangle([0,0,400,400], fill="black",outline="black")
    cv.create_rectangle(0,0,400,400,fill="black",outline="black")


lable=tk.Label(fr,text="Number: ",fg="black",font=("Helvetica", 18))
lable.place(x=350,y=20)
cb=tk.Button(fr, command= clear,image=pixel, width=100, height=50, text="Clear",compound="c")
cb.place(x=350,y=100)

def check():
    global img
    global dimg
    img = img.resize((28, 28), Image.ANTIALIAS)
    img.save("mydrawing.png")
    test = imageio.imread("mydrawing.png")

    iminp = np.ones((28, 28))
    for r in range(test.shape[0]):
        for c in range(test.shape[1]):
            iminp[r][c] = test[r][c][0]
    iminp = iminp.reshape((784, 1)) / 255.0

    temp = nn.output(iminp)
    max1 = temp[0][0]
    ind = 0
    for i in range(9):
        if (max1 < temp[i + 1][0]):
            ind = i + 1
            max1 = temp[i + 1][0]

    lable.configure(text="Number: "+str(ind))
    img = Image.new("RGB", (w1, h1), color="black")
    dimg = ImageDraw.Draw(img)


chck=tk.Button(fr, command = check,image=pixel, width=100, height=50, text="Check",compound="c")
chck.place(x=350,y=180)

def callback(event):
    global draw
    global brush_size
    draw=True

    plist=[(event.x-brush_size,event.y-brush_size),(event.x+brush_size,event.y+brush_size)]
    dimg.ellipse(plist,fill=(255,255,255))
    cv.create_oval(event.x-brush_size,event.y-brush_size,event.x+brush_size,event.y+brush_size, fill="white",outline="white")
def callback1(event):
    global draw
    global brush_size
    draw=False
    plist = [(event.x - brush_size, event.y - brush_size), (event.x + brush_size, event.y + brush_size)]
    dimg.ellipse(plist, fill=(255, 255, 255))
    cv.create_oval(event.x-brush_size,event.y-brush_size,event.x+brush_size,event.y+brush_size, fill="white",outline="white")
def callback2(event):
    global draw
    global brush_size
    if(draw):
        plist = [(event.x - brush_size, event.y - brush_size), (event.x + brush_size, event.y + brush_size)]
        dimg.ellipse(plist, fill=(255, 255, 255))
        cv.create_oval(event.x - brush_size, event.y - brush_size, event.x + brush_size, event.y + brush_size, fill="white", outline="white")


cv.bind("<Button-1>",callback)
cv.bind("<ButtonRelease-1>",callback1)
cv.bind("<Motion>",callback2)

cv.focus_set()
cv.create_line(0,0,50,50)
root.mainloop()
