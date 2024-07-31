import ghhops_server as hs
from flask import Flask
import rhino3dm as rh
import joblib

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation
import json
import numpy as np
import pandas as pd
import math
import copy
import os
import re


loaded_model = joblib.load('gradient_boosting_model.joblib')
files = os.listdir("PredictionModels\\")
model_dict = {}
for file in files:
    name = re.split(r'\.',file)[0]
    model = joblib.load('PredictionModels\\{}'.format(file))
    model_dict[name] = model

app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)



def build_frames(data):
    frames = []
    for i in range(len(data["X"])):
        frame = [data["X"][i],data["Y"][i],data["Z"][i],data["A"][i],data["B"][i],data["C"][i]]
        frames.append(frame)
    return frames

def distance(f1,f2):
    dist = math.sqrt((f2[0]-f1[0])**2 + (f2[1]-f1[1])**2 +(f2[2]-f1[2])**2)
    return dist 

def angle_difference(frame1, frame2):
    # Extract Euler angles from each frame
    euler_angles1 = [float(x) for x in frame1[3:]]
    euler_angles2 = [float(x) for x in frame2[3:]]

    # Create rotation matrices from Euler angles
    rotation_matrix1 = Rotation.from_euler('xyz', euler_angles1, degrees=True).as_matrix()
    rotation_matrix2 = Rotation.from_euler('xyz', euler_angles2, degrees=True).as_matrix()

    # Calculate the rotation matrix that transforms frame1 to frame2
    relative_rotation_matrix = rotation_matrix2 @ rotation_matrix1.T

    # Convert the rotation matrix to axis-angle representation
    axis_angle = Rotation.from_matrix(relative_rotation_matrix).as_rotvec()

    # Calculate the angle difference
    angle_difference = np.linalg.norm(axis_angle)

    return angle_difference

def calculateTaskDistance(frames,frames2,currentPos):
    
    dist = 0
    angle = 0
    #this is glue task 
    if frames != None and frames2 != None and frames!=[] and frames2!=[]:
        #frames = re.split(';',frames)
        #frames2 = re.split(';',frames2)
                
        dist += distance(frames[0],currentPos)
        angle += abs(angle_difference(currentPos,frames[0]))
        
        if len(frames)==1:
            #print ("angle---",angle)
            return dist,angle,frames2[-1]
        
        for i in range(1,len(frames)):
            dist += distance(frames[i-1],frames2[i-1])
            dist += distance(frames2[i-1],frames[i])
            
            angle += abs(angle_difference(frames[i-1],frames2[i-1]))
            angle += abs(angle_difference(frames2[i-1],frames[i]))
            
            
            
        dist += distance(frames[-1],frames2[-1])
        angle += abs(angle_difference(frames[-1],frames2[-1]))
        finalPos=frames2[-1]
        #print ("angle---",angle)
        return dist,angle,finalPos
        
    #this is a regular task
    elif frames != None and frames!=[]:
        
        #frames = re.split(';',frames)
        
        dist += distance(frames[0],currentPos)
        angle += abs(angle_difference(frames[0],currentPos))
        
        if len(frames)==1: 
            #print ("angle---",angle)
            return dist,angle,frames[0]
            
        for i in range(1,len(frames)):
            dist += distance(frames[i-1],frames[i])
            angle += abs(angle_difference(frames[i-1],frames[i]))
    
        #print ("angle---",angle)
        return dist,angle,frames[-1]
    
    #print ("angle---",angle)
    return dist,angle,currentPos







# Define the Objective functions
# Objective Functions from https://machinelearningmastery.com/curve-fitting-with-python/
def objective(x, a, b):
    return a * x + b

def objective2(x, a, b, c):
    return a * x + b * x**2 + c

def objective3(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def fit_to_objective(xdata,ydata,obj_fun = 0):

    if obj_fun==0:
        parameters, _ = curve_fit(objective, xdata, ydata)
        a,b = parameters
        x_line = np.arange(min(xdata), max(xdata), 1)
        y_line = objective(x_line, a, b)
    elif obj_fun==1:
        parameters, _ = curve_fit(objective2, xdata, ydata)
        a,b,c = parameters
        x_line = np.arange(min(xdata), max(xdata), 1)
        y_line = objective2(x_line, a, b, c)
    elif obj_fun==2:
        parameters, _ = curve_fit(objective3, xdata, ydata)
        a, b, c, d, e, f = parameters
        x_line = np.arange(min(xdata), max(xdata), 1)
        y_line = objective3(x_line, a, b, c,d,e,f)
    else:
        x_line,y_line = [],[]
    return x_line,y_line

@app.route("/index")
def index():
    return ("This is my Flask app")

@hops.component(
    "/sum",
    name="Sum",
    description="sum of numbers",
    inputs=[
        hs.HopsNumber("A", "A", "First Number"),
        hs.HopsNumber("B", "B", "Second Number")
    ],
    outputs=[
        hs.HopsNumber("Sum", "Sum", "Sum of the numbers")
    ]
)
def sum (a:float, b:float):
    return a+b

@hops.component(
    "/plotFit",
    name = "Plt",
    description = "This component plot the point and save the png files",
    inputs = [hs.HopsBoolean("Plot","Plot","True to Plot",access=hs.HopsParamAccess.ITEM),
              hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.LIST),
              hs.HopsCurve("Curve","Curve","Fit curve to plot",access=hs.HopsParamAccess.ITEM),
              hs.HopsString("File Name","File Name","File Name",access=hs.HopsParamAccess.ITEM)
    ],
    outputs=[hs.HopsBoolean("Success","Success","True if plotted False if error",access=hs.HopsParamAccess.ITEM)
    ]
)
def plotFit(save,pts,crv,file_name):
    X = [pt.X for pt in pts]
    Y = [pt.Y for pt in pts]

    crv.Domain = rh.Interval(0.00,1.00)
    cX = []
    cY = []
    for i in range(len(pts)):
        pt = crv.PointAt(i/len(pts))
        cX.append(pt.X)
        cY.append(pt.Y)
    if save:
        fig,ax = plt.subplots()
        ax.plot(X, Y, 'o', label='data')
        ax.plot(cX, cY, '-', label='fit')
        ax.legend()
        name = "{}.png".format(file_name)
        fig.savefig(name)
        return True
    
    return False

@hops.component(
    "/fitCurve",
    name = "fit",
    description = "This component fit a curve to a set of points",
    inputs = [hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.LIST),
              hs.HopsNumber("Objective","Obj","Objective Type",access=hs.HopsParamAccess.ITEM)
    ],
    outputs=[hs.HopsBoolean("Success","Success","True if fitted curve",access=hs.HopsParamAccess.ITEM),
            hs.HopsCurve("fitCurve","CRV","Fitted curve",access=hs.HopsParamAccess.ITEM)
    ]
)
def fitCurve(pts,obj):
    X = [pt.X for pt in pts]
    Y = [pt.Y for pt in pts]
    Z = [pt.Z for pt in pts]

    xdata = np.asarray(X)
    ydata = np.asarray(Y)

    x_line,y_line = fit_to_objective(xdata,ydata,obj_fun=obj)

    newPoints = []
    if len(x_line)>0:
        for i in range(len(x_line)):
            pt = rh.Point3d(x_line[i],y_line[i],0)
            newPoints.append(pt)
        crv = rh.Curve.CreateControlPointCurve(newPoints,1)
        print ("I'm here now")
        return True,crv
    print ("lol")
    crv = rh.Curve.CreateControlPointCurve(pts,1)
    print (crv)
    return False,crv

@hops.component(
    "/savePoints",
    name = "savePoints",
    description = "This component save a list of points as json file",
    inputs = [hs.HopsPoint("Points","Points","Points to plot",access=hs.HopsParamAccess.LIST)
    ],
    outputs=[hs.HopsBoolean("Success","Success","True if plotted False if error",access=hs.HopsParamAccess.ITEM)
    ]
)
def savePoints(pts):
    X = [pt.X for pt in pts]
    Y = [pt.Y for pt in pts]
    Z = [pt.Z for pt in pts]
    
    points_dict = {'X':X,'Y':Y,'Z':Z}
    
    file_path = 'points.json'


    with open(file_path, 'w') as json_file:
        json.dump(points_dict, json_file, indent=4)

    print(f"JSON file saved at: {file_path}")
        
    return True

@hops.component(
    "/predictPoint",
    name = "predictPoint",
    description = "This component predict a z value of a point on a surface",
    inputs = [hs.HopsPoint("Point","Point","Points to test",access=hs.HopsParamAccess.ITEM)
    ],
    outputs=[hs.HopsPoint("Point","Point","Point with predicted Z value",access=hs.HopsParamAccess.ITEM)
    ]
)
def predictPoint(pt):
    X = pt.X 
    Y = pt.Y
    new_data = np.array([[X,Y]])
    prediction = loaded_model.predict(new_data)
    new_pt = rh.Point3d(X,Y,prediction)
    return new_pt


@hops.component(
    "/tester",
    name = "Tester",
    description = "None",
    inputs = [hs.HopsString("data","data","data",access=hs.HopsParamAccess.LIST),
              hs.HopsString("model_name","model","model",access=hs.HopsParamAccess.ITEM)],
    outputs = [hs.HopsNumber("data","data","data",access=hs.HopsParamAccess.LIST)]
)
def tester(data,model_name):

    new_data = []
    
    HOME = [2316.51,0.08,2961.99,-89.81,-44.85,-90.26]
    current_pos = copy.deepcopy(HOME)
    for d in data:
        data_dict = json.loads(d)
        
        name = data_dict["type"]
        if "clos" in name.lower() : name="Closing"
        if "init" in name.lower():name = "Initiate Tool"
        if "place" in name.lower(): name = "Place"
        
        new_dict = {"names":name}
        actor_data = data_dict["actors_data"][data_dict["main_actor"]]
        
        new_dict["tools"] = actor_data["toolname"]
        new_dict["operationCount"] = actor_data["actioncount"]
        
        
        if actor_data["movementbasetype"] == "axis":
            frames = build_frames(actor_data["secondaryactionvalues"])
            dist,angle,current_pos = calculateTaskDistance(frames,None,current_pos)
        else:
            if actor_data["actionid"]==1:
                dist,angle,current_pos = calculateTaskDistance([HOME],None,current_pos)
            elif actor_data["mainactionvalues"]==None:
                dist,angle,current_pos = 0,0,current_pos  
            elif actor_data["secondaryactionvalues"] != None:
                frames,frames2 = build_frames(actor_data["mainactionvalues"]),build_frames(actor_data["secondaryactionvalues"])
                dist,angle,current_pos = calculateTaskDistance(frames,frames2,current_pos)
            else:
                frames = build_frames(actor_data["mainactionvalues"])
                dist,angle,current_pos = calculateTaskDistance(frames,None,current_pos)
        
        dist = dist/10000
        new_dict["subOperationDistance"] = dist/actor_data["actioncount"]
        new_dict["totalDistance"] = dist
        new_dict["movementAngle"] = angle
        
        new_data.append(new_dict)
    
        
    data_dict = {key: []  for key, value in new_data[0].items()}
    
    for d in new_data:
        dict = d
        for key,value in dict.items():
            data_dict[key].append(value)
    
    tool_map = {'Nail Gripper':1, 'Vaccuum Gripper':2, 'GlueGun':3}
    task_map = {'Closing':1,'Gluing':2,'Initiate Tool':3,'Move':4,'Nail':5,'Pick':6,'Place':7,
                'Spilling':8,'Storing':9,'Take Tool':10,'Travel':11}
    
    df = pd.DataFrame(data_dict)
    df = df.map(lambda s: tool_map.get(s) if s in tool_map else s)
    df = df.map(lambda s: task_map.get(s) if s in task_map else s)
    
    predictions = model_dict[model_name].predict(df)
    results = [float(p) for p in predictions]
    #print (results)
    return (results)
    
    
    return new_data

@hops.component(
    "/predict_tester",
    name = "PredictTester",
    description = "None",
    inputs = [hs.HopsString("data","data","data",access=hs.HopsParamAccess.LIST),
              hs.HopsString("model_name","model","model",access=hs.HopsParamAccess.ITEM)],
    outputs = [hs.HopsNumber("data","data","data",access=hs.HopsParamAccess.LIST)]
)
def predict_tester(data,model_name):

    tool_map = {'Nail Gripper':1, 'Vaccuum Gripper':2, 'GlueGun':3}
    task_map = {'Closing':1,'Gluing':2,'Initiate Tool':3,'Move':4,'Nail':5,'Pick':6,'Place':7,
                'Spilling':8,'Storing':9,'Take Tool':10,'Travel':11}
    
    dict = json.loads(data[0])
    data_dict = {key: []  for key, value in dict.items()}
    
    for d in data:
        dict = json.loads(d)
        for key,value in dict.items():
            data_dict[key].append(value)
    
    df = pd.DataFrame(data_dict)
    df = df.map(lambda s: tool_map.get(s) if s in tool_map else s)
    df = df.map(lambda s: task_map.get(s) if s in task_map else s)
    
    predictions = model_dict[model_name].predict(df)
    results = [float(p) for p in predictions]
    #print (results)
    return (results)
    
    
        


if __name__ == "__main__":

    app.run(debug=True)



