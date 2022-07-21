from flask import Flask,render_template,request, flash,redirect, url_for, jsonify
#from config import client
#from models import *
#import secrets
import datetime
import json
#from flask_mail import Mail,Message
#from bson import ObjectId
import os
import numpy as np
import urllib.request
import os
#from werkzeug.utils import secure_filename
import cv2
#from flask_mysqldb import MySQL
#from flaskext.mysql import MySQL
#import pymysql
import requests
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import cv2 as cv

app = Flask(__name__)
app.super_secret_key='irisproject'
#app.config['MYSQL_HOST'] = 'scanner-iris-db.cj1kyfpqy43v.us-east-2.rds.amazonaws.com'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = 'Varsha123'
#app.config['MYSQL_DB'] = 'retina_scanner'
#app.config['MYSQL_CURSORCLASS']='DictCursor'
#app.config['MYSQL_CONNECT_TIMEOUT']=180
#app.config['MYSQL_PORT']=3306
#mysql = MySQL(app)
import pymysql

mysql = pymysql.connect(host='scanner-iris-db.cj1kyfpqy43v.us-east-2.rds.amazonaws.com', user="root", passwd="Varsha123", database="retina_scanner")
data_status = {"responseStatus": 0, "result": ""}


@app.route("/employeeAuth",methods=['POST','GET'])

def irisMacthing():
    def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
        orb = cv2.ORB_create()

        # 714 x 901 pixels

        # detect keypoints and descriptors
        kp_a, desc_a = orb.detectAndCompute(img1, None)
        kp_b, desc_b = orb.detectAndCompute(img2, None)

        # define the bruteforce matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # perform matches.
        matches = bf.match(desc_a, desc_b)
        # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
        similar_regions = [i for i in matches if i.distance < 60]
        if len(matches) == 0:
            return 0
        return len(similar_regions) / len(matches)
    def iris(path):
        #img = cv.imread(path)
        img=path
        output = img.copy()
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(img, 5)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=122, maxRadius=138)
        detected_circles = np.uint16(np.around(circles))
        for (x, y ,r) in detected_circles[0, :]:
            cv.circle(output, (x, y), r, (0, 255, 0), 3)
            cv.circle(output, (x, y), 2, (0, 255, 255), 3)

        if detected_circles is not None:
            circle = detected_circles[0][0]
            iris_coordinates = (circle[0], circle[1])
        #print(iris_coordinates)

        if iris_coordinates is not None:
            x = int(iris_coordinates[0])
            y = int(iris_coordinates[1])

            w = int(round(circle[2]) + 10)
            h = int(round(circle[2]) + 10)

        #cv2.circle(original_eye, iris_coordinates, int(circle[2]), (255,0,0), thickness=
            iris_image = img[y - h:y + h, x - w:x + w]
            iris_image_to_show = cv.resize(iris_image, (iris_image.shape[1] * 2, iris_image.shape[0] * 2))

        q = np.arange(0.00, np.pi * 2, 0.01)  # theta
        inn = np.arange(0, int(iris_image_to_show.shape[0] / 2), 1)  # radius

        cartisian_image = np.empty(shape=[inn.size, int(iris_image_to_show.shape[1]), 3])
        m = interp1d([np.pi * 2, 0], [0, iris_image_to_show.shape[1]])

        for r in inn:
            for t in q:
                polarX = int((r * np.cos(t)) + iris_image_to_show.shape[1] / 2)
                polarY = int((r * np.sin(t)) + iris_image_to_show.shape[0] / 2)
                cartisian_image[r][int(m(t) - 1)] = iris_image_to_show[polarY][polarX]

        im = Image.fromarray(iris_image_to_show)
        return im

    data_status = {"responseStatus": 0, "result": ""}

    id = request.form.get("userId")

    lat = request.form.get("lat")
    long = request.form.get("long")
    attendType = request.form.get("attendType")
    try:
        try:

            eyeImage = request.files["eyeImage"]

            npimg = np.fromfile(eyeImage, np.uint8)

            file3 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            file3 = cv2.cvtColor(file3, cv.COLOR_BGR2GRAY)

            # file3=np.array(iris(file3))

        except KeyError:
            data_status["responseStatus"] = 0
            data_status["codeError"] = 400
            data_status["result"] = "KeyError"
            return data_status
    except cv2.error as e:
        data_status["responseStatus"] = 0
        data_status["code"] = 402
        data_status["result"] = "Select image file"
        return data_status

    if request.method == "POST":

        query = "select left_eye,right_eye from retina_scanner.user where id={}".format(id)
        cur = mysql.cursor()
        cur.execute(query)
        data = cur.fetchall()
        temp = 0
        for i in data:
            data_list = []
            print(i)
            for n in i:
                data_list.append(n)
            for d in data_list:

                try:

                    response = requests.get(d).content
                    print(data_list)

                    img = plt.imread(io.BytesIO(response), format='bmp')

                    # img=np.array(iris(img))
                    print("orb", d)

                    orb_similarity = orb_sim(img, file3)
                    print(orb_similarity)
                    if orb_similarity > 0.35:
                        query1 = "select id, first_name, last_name, phone, image, aadhar from retina_scanner.user where id={}".format(
                            id)
                        # cursor=conn.cursor()
                        cur.execute(query)
                        img = cur.fetchall()
                        cur.execute(query1)
                        # print(img)
                        img2 = cur.fetchall()
                        # print(img2)
                        sim = 0
                        emplist = []
                        emp_dict = {}
                        for i in img[0]:
                            emplist.append(i)

                        in_time = datetime.datetime.now()
                        user_id = id
                        if attendType == "in":

                            cur.execute("INSERT INTO retina_scanner.user_login_info values(%s,%s, %s, %s, %s,%s,%s,%s)",
                                        (None, user_id, in_time, lat, long, None, None, None))
                            mysql.commit()
                            print(img2)
                            emp_dict["id"] = img2[0][0]
                            emp_dict["first_name"] = img2[0][1]
                            emp_dict["last_name"] = img2[0][2]
                            emp_dict["phone"] = img2[0][3]
                            emp_dict["image"] = img2[0][4]
                            emp_dict["aadhar"] = img2[0][5]
                            emp_dict["Time"] = datetime.datetime.now()

                            data_status["responseStatus"] = 1
                            data_status["statusCode"] = 200
                            data_status["status"] = "sucess"
                            data_status["result"] = "Matched"
                            data_status["orbSimilarity"] = orb_similarity
                            data_status["imageName"] = d[:-4].split("/")[-1]
                            data_status["id"] = id
                            data_status["employeeInfo"] = emp_dict
                            if d == data_list[1]:
                                data_status["matchFrom"] = "Image from Left eye"
                            else:
                                data_status["matchFrom"] = "Image from Right eye"

                        elif attendType == "out":
                            out_time = datetime.datetime.now()
                            cur.execute(
                                "update retina_scanner.user_login_info set out_time_attend=%s ,out_time_lat=%s,out_time_long=%s where fk_user_id=%s and date(in_time_attend)=date(now())",
                                (out_time, lat, long, user_id))

                            mysql.commit()
                            emp_dict["id"] = img2[0][0]
                            emp_dict["first_name"] = img2[0][1]
                            emp_dict["last_name"] = img2[0][2]
                            emp_dict["phone"] = img2[0][3]
                            emp_dict["image"] = img2[0][4]
                            emp_dict["aadhar"] = img2[0][5]
                            emp_dict["Time"] = datetime.datetime.now()

                            data_status["responseStatus"] = 1
                            data_status["statusCode"] = 200
                            data_status["status"] = "sucess"
                            data_status["result"] = "Matched"
                            data_status["orbSimilarity"] = orb_similarity
                            data_status["imageName"] = d[:-4].split("/")[-1]
                            data_status["id"] = id
                            data_status["employeeInfo"] = emp_dict

                            if d == data_list[1]:
                                data_status["matchFrom"] = "Image from Left eye"
                            else:
                                data_status["matchFrom"] = "Image from Right eye"

                        else:
                            data_status["responseStatus"] = 0
                            data_status["statusCode"] = 405
                            data_status["result"] = "Select 'in' or 'out' option only"

                        return data_status

                    else:
                        if orb_similarity > temp:
                            temp = orb_similarity
                        continue
                except cv2.error as e:
                    data_status["responseStatus"] = 0
                    data_status["code"] = 403
                    data_status["result"] = "Select Image format"
                return data_status

            else:
                data_status["responseStatus"] = 0
                data_status["statusCode"] = 422
                data_status["status"] = "failed"
            # data_status["orbsimilarity"]=temp
                data_status["result"] = "File Not Match"
                data_status["ID"] = 0
            return data_status

@app.route("/managerAuth",methods=['POST','GET'])

def Managerloginpage():
    def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
        orb = cv2.ORB_create()

        # 714 x 901 pixels

        # detect keypoints and descriptors
        kp_a, desc_a = orb.detectAndCompute(img1, None)
        kp_b, desc_b = orb.detectAndCompute(img2, None)

        # define the bruteforce matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # perform matches.
        matches = bf.match(desc_a, desc_b)
        # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
        similar_regions = [i for i in matches if i.distance < 60]
        if len(matches) == 0:
            return 0
        return len(similar_regions) / len(matches)
    def iris(path):
        #img = cv.imread(path)
        img=path
        output = img.copy()
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(img, 5)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=122, maxRadius=138)
        detected_circles = np.uint16(np.around(circles))
        for (x, y ,r) in detected_circles[0, :]:
            cv.circle(output, (x, y), r, (0, 255, 0), 3)
            cv.circle(output, (x, y), 2, (0, 255, 255), 3)

        if detected_circles is not None:
            circle = detected_circles[0][0]
            iris_coordinates = (circle[0], circle[1])
        #print(iris_coordinates)

        if iris_coordinates is not None:
            x = int(iris_coordinates[0])
            y = int(iris_coordinates[1])

            w = int(round(circle[2]) + 10)
            h = int(round(circle[2]) + 10)

        #cv2.circle(original_eye, iris_coordinates, int(circle[2]), (255,0,0), thickness=
            iris_image = img[y - h:y + h, x - w:x + w]
            iris_image_to_show = cv.resize(iris_image, (iris_image.shape[1] * 2, iris_image.shape[0] * 2))

        q = np.arange(0.00, np.pi * 2, 0.01)  # theta
        inn = np.arange(0, int(iris_image_to_show.shape[0] / 2), 1)  # radius

        cartisian_image = np.empty(shape=[inn.size, int(iris_image_to_show.shape[1]), 3])
        m = interp1d([np.pi * 2, 0], [0, iris_image_to_show.shape[1]])

        for r in inn:
            for t in q:
                polarX = int((r * np.cos(t)) + iris_image_to_show.shape[1] / 2)
                polarY = int((r * np.sin(t)) + iris_image_to_show.shape[0] / 2)
                cartisian_image[r][int(m(t) - 1)] = iris_image_to_show[polarY][polarX]

        im = Image.fromarray(iris_image_to_show)
        return im

    Username = request.form.get("Username")
    Password = request.form.get("Password")
    lat = request.form.get("lat")
    long = request.form.get("long")
    attendType = request.form.get("attendType")
    data_status = {"responseStatus": 0, "result": ""}
    if Username and Password and request.method == "POST":
        cur = mysql.cursor()
        query = "SELECT * FROM retina_scanner.manager_login;"
        cur.execute(query)
        data = cur.fetchall()
        print(data)
        for i in data:
            dummy = ""
            if i[2] == Username:
                if i[3] == Password:
                    user_id = i[1]
                    print(i[1], i[0])
                    break
                else:
                    dummy = "not"
            else:
                dummy = "not"
        if dummy == "not":

            data_status["responseStatus"] = 0
            data_status["statusCode"] = 401
            data_status["result"] = "Credentials failed"
            data_status["status"] = "Credentials failure"
            data_status["currentTime"] = datetime.datetime.now()

        else:
            eyeImage = request.files["eyeImage"]
            npimg = np.fromfile(eyeImage, np.uint8)
            # print(eyeScanImage)
            obj = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            obj = cv.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            # obj=np.array(iris(obj))
            # print(obj)
            query = "select left_eye,right_eye from retina_scanner.user where id={}".format(user_id)
            query1 = "select id, first_name, last_name, phone, image, aadhar from retina_scanner.user where id={}".format(
                user_id)
            # cursor=conn.cursor()
            cur.execute(query)
            img = cur.fetchall()
            cur.execute(query1)
            # print(img)
            img2 = cur.fetchall()
            sim = 0
            emplist = []
            print("images", img, user_id)
            for i in img[0]:
                emplist.append(i)
                # print(i.items())
                # for j in i.items():
                # emplist.append(j[1])
            for d in emplist:
                # print(d)

                temp = ""
                response = requests.get(d).content
                img1 = plt.imread(io.BytesIO(response), format='bmp')
                # img1=np.array(iris(img1))

                orb_similarity = orb_sim(img1, obj)
                emp_dict = {}
                print(orb_similarity)
                if orb_similarity > 0.35:
                    in_time = datetime.datetime.now()
                    if attendType == "in":
                        cur.execute("SELECT fk_user_id, in_time_attend from retina_scanner.user_login_info")
                        emp_data = cur.fetchall()
                        # print(emp_data)
                        not_found = ""
                        for i in range(len(emp_data)):
                            not_found = ""

                            if emp_data[i][0] == user_id and emp_data[i][1].strftime("%Y-%m-%d") == in_time.strftime(
                                    "%Y-%m-%d"):
                                cur.execute(
                                    "update retina_scanner.user_login_info set in_time_attend=%s ,in_time_lat=%s,in_time_long=%s where fk_user_id=%s and date(in_time_attend)=date(now())",
                                    (in_time, lat, long, user_id))
                            else:
                                not_found = "not"
                        if not_found == "not":
                            cur.execute("INSERT INTO retina_scanner.user_login_info values(%s,%s, %s, %s, %s,%s,%s,%s)",
                                        (None, user_id, in_time, lat, long, None, None, None))
                        mysql.commit()
                        print(img2)
                        emp_dict["id"] = img2[0][0]
                        emp_dict["first_name"] = img2[0][1]
                        emp_dict["last_name"] = img2[0][2]
                        emp_dict["phone"] = img2[0][3]
                        emp_dict["image"] = img2[0][4]
                        emp_dict["aadhar"] = img2[0][5]
                        emp_dict["Time"] = datetime.datetime.now()
                        data_status["responseStatus"] = 1
                        data_status["statusCode"] = 200
                        data_status["result"] = "success"
                        data_status["orbSimilarity"] = orb_similarity
                        data_status["imageName"] = d[:-4].split("/")[-1]
                        if d == emplist[0]:

                            data_status["matchFrom"] = "Image from Left eye"
                        else:
                            data_status["matchFrom"] = "Image from Right eye"
                        data_status["employeeInfo"] = emp_dict
                    elif attendType == "out":
                        out_time = datetime.datetime.now()
                        cur.execute(
                            "update retina_scanner.user_login_info set out_time_attend=%s ,out_time_lat=%s,out_time_long=%s where fk_user_id=%s and date(in_time_attend)=date(now())",
                            (out_time, lat, long, user_id))
                        mysql.connection.commit()
                        emp_dict["id"] = img2[0]["id"]
                        emp_dict["first_name"] = img2[0]["first_name"]
                        emp_dict["last_name"] = img2[0]["last_name"]
                        emp_dict["phone"] = img2[0]["phone"]
                        emp_dict["image"] = img2[0]["image"]
                        emp_dict["aadhar"] = img2[0]["aadhar"]
                        emp_dict["Time"] = datetime.datetime.now()

                        data_status["responseStatus"] = 1
                        data_status["statusCode"] = 200
                        data_status["result"] = "success"
                        data_status["orbSimilarity"] = orb_similarity
                        data_status["imageName"] = d[:-4].split("/")[-1]
                        data_status["employeeInfo"] = emp_dict
                        if d == emplist[0]:
                            data_status["matchFrom"] = "Image from Left eye"
                        else:
                            data_status["matchFrom"] = "Image from Right eye"
                    else:
                        data_status["responseStatus"] = 0
                        data_status["statusCode"] = 405
                        data_status["result"] = "Select 'in' or 'out' option only"

                    break
                else:
                    if sim < orb_similarity:
                        sim = orb_similarity
                    temp = "not found"
                if temp == "not found":
                    user_dict = {}
                    user_dict["username"] = Username
                    user_dict["orbSimilarity"] = sim
                    user_dict["currentTime"] = datetime.datetime.now()

                    data_status["responseStatus"] = 0
                    data_status["statusCode"] = 403
                    data_status["result"] = "Eye scan not matched"
                    # data_status["orbSimilarity"]=orb_similarity
                    data_status["employeeInfo"] = user_dict

            return data_status

@app.route("/managerlogout",methods=['POST','GET'])

def Managerlogout():
    def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
        orb = cv2.ORB_create()

        # 714 x 901 pixels

        # detect keypoints and descriptors
        kp_a, desc_a = orb.detectAndCompute(img1, None)
        kp_b, desc_b = orb.detectAndCompute(img2, None)

        # define the bruteforce matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # perform matches.
        matches = bf.match(desc_a, desc_b)
        # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
        similar_regions = [i for i in matches if i.distance < 60]
        if len(matches) == 0:
            return 0
        return len(similar_regions) / len(matches)

    id = request.form.get("userId")
    lat = request.form.get("lat")
    long = request.form.get("long")
    eyeImage = request.files["eyeImage"]
    data_status = {"responseStatus": 0, "result": ""}
    npimg = np.fromfile(eyeImage, np.uint8)
    # print(eyeScanImage)
    obj = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # print(obj)
    # query ="select left_eye,right_eye from retina_scanner.user where id={}".format(id)
    if request.method == "POST":
        qry = "select id from retina_scanner.user"
        cur = mysql.cursor()
        cur.execute(qry)
        ids1 = cur.fetchall()
        ids = []
        for i in ids1:
            ids.append(i[0])
        #print(ids)

        if int(id) in ids:
            #print(id)
            query = "select left_eye,right_eye from retina_scanner.user where id={}".format(id)
            cur = mysql.cursor()
            cur.execute(query)
            data = cur.fetchall()
            emplist = []
            for i in data[0]:
                emplist.append(i)
            #print(data)
            for d in emplist:
                temp = ""
                response = requests.get(d).content
                img1 = plt.imread(io.BytesIO(response), format='bmp')
                orb_similarity = orb_sim(img1, obj)
                # emp_dict={}
                #print(orb_similarity)
                if orb_similarity > 0.20:
                    time = datetime.datetime.now()
                    cur.execute(
                        "update retina_scanner.user_login_info set out_time_attend=%s ,out_time_lat=%s,out_time_long=%s where fk_user_id=%s and date(in_time_attend)=date(now())",
                        (time, lat, long, id))
                    data_status["Response_status"] = 1
                    data_status["statusCode"] = 200
                    data_status["result"] = "succusessfull"
                    break
                else:
                    temp = "not found"
                    continue
            if temp == "not found":

                data_status["response_status"] = 0
                data_status["statusCode"] = 422
                data_status["result"] = "file not match"
                data_status["status"] = "failed"
        else:
            data_status["responseStatus"] = 0
            data_status["statusCode"] = 423
            data_status["result"] = "enter valid id"

    return data_status
@app.route("/employeexist",methods=['POST','GET'])

def employeexist():
    def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
        orb = cv2.ORB_create()

        # 714 x 901 pixels

        # detect keypoints and descriptors
        kp_a, desc_a = orb.detectAndCompute(img1, None)
        kp_b, desc_b = orb.detectAndCompute(img2, None)

        # define the bruteforce matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # perform matches.
        matches = bf.match(desc_a, desc_b)
        # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
        similar_regions = [i for i in matches if i.distance < 60]
        if len(matches) == 0:
            return 0
        return len(similar_regions) / len(matches)

    leftEyeImage = request.files["leftEyeImage"]
    npimg = np.fromfile(leftEyeImage, np.uint8)
    # print(eyeScanImage)
    lefteye = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rightEyeImage = request.files["rightEyeImage"]
    npimg = np.fromfile(rightEyeImage, np.uint8)
    # print(eyeScanImage)
    righteye = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    data_status = {"responseStatus": 0, "result": ""}
    if request.method == "POST":
        cur = mysql.cursor()
        query = "SELECT left_eye,right_eye FROM retina_scanner.user;"
        cur.execute(query)
        data = cur.fetchall()
        for i in data:
            temp = ""
            response1 = requests.get(i[0]).content
            left = plt.imread(io.BytesIO(response1), format='bmp')

            orb_similarity1 = orb_sim(left, lefteye)
            response2 = requests.get(i[1]).content
            right = plt.imread(io.BytesIO(response2), format='bmp')

            orb_similarity2 = orb_sim(right, righteye)
            if (orb_similarity1 > 0.95) and (orb_similarity2 > 0.95):
                data_status["status"] = "succsess"
                data_status["responseStatus"] = 1
                data_status["result"] = "employee exist"
                data_status["statuscode"] = 200
                break
            else:
                temp = "not"
                continue
        if temp == "not":
            # data_status["responsestatus"]=0
            data_status["statuscode"] = 422
            data_status["result"] = "employee does not exist"

    return data_status

@app.route("/all_users",methods=["POST","GET"])
def all_users():
    data_status={"responseStatus":0,"result":""}
    if request.method=="GET":
        query="select * from retina_scanner.user;"
        cursor=mysql.cursor()
        cursor.execute(query)
        data=cursor.fetchall()
        data1=list(data)

        employee=[]
        for i in data:
            dic={}
            dic["id"]=i[0]
            dic["first_name"]=i[1]
            dic["last_name"]=i[2]
            dic["phone"]=i[3]
            dic["email"]=i[4]
            dic["gender"]=i[5]
            dic["age"]=i[6]
            dic["aadhar"]=i[10]
            dic["address"]=i[13]
            dic["village"]=i[16]
            dic["district"]=i[15]
            dic["state"]=i[14]
            dic["image"]=i[18]
            dic["left_eye"]=i[19]
            dic["right_eye"]=i[20]
            dic["type"]=i[21]
            dic["fk_manager_id"]=i[22]
            employee.append(dic)
        data_status["Employee_list"]=employee
        return data_status


@app.route("/getemployeejobs", methods=["POST", "GET"])
def setemployeejobs():
    data_status = {"responseStatus": 0, "result": ""}
    employee_id = request.form["employee_id"]
    manager_id = request.form["manager_id"]
    if request.method == "POST":
        cur = mysql.cursor()
        cur.execute("select fk_employee_id,fk_manager_id from retina_scanner.employee_job_details")
        ids = cur.fetchall()
        # emp_id=[]
        # mng_id=[]
        # manager_id=16
        print(ids)

        for i in ids:

            n = "not"

            list1 = []
            # emp_id.append(i["fk_employee_id"])
            # mng_id.append(i["fk_manager_id"])

            if i[0] == int(employee_id) and i[1] == int(manager_id):

                cur = mysql.cursor()
                cur.execute(
                    "SELECT fk_job_id,date(start_date),date(end_date) FROM retina_scanner.employee_job_details where fk_employee_id=%s and fk_manager_id=%s and completion=0",
                    (employee_id, manager_id))
                d = cur.fetchall()

                for i in d:
                    d1 = []
                    for j in i:
                        d1.append(j)
                    # for i in d1:

                    dict1 = {}
                    cur.execute(
                        "SELECT job_name,job_description,state,district,block,panchayat,sector,job_duration_in_days FROM retina_scanner.job_details where job_id={}".format(
                            d1[0]))
                    job = cur.fetchall()
                    d2 = []

                    for i in job[0]:
                        d2.append(i)

                    dict1["job_id"] = d1[0]
                    dict1["start_date"] = d1[1].strftime("%b %d %Y")
                    dict1["end_date"] = d1[2].strftime("%b %d %Y")
                    dict1["job_name"] = d2[0]
                    dict1["job_description"] = d2[1]
                    dict1["state"] = d2[2]
                    dict1["district"] = d2[3]
                    dict1["block"] = d2[4]
                    dict1['panchayat'] = d2[5]
                    dict1['sector'] = d2[6]
                    dict1["job_duration_in_days"] = d2[7]

                    list1.append(dict1)
                    # print(list1)
                data_status["details"] = list1
                data_status["responseStatus"] = 1
                data_status["statuscode"] = 200
                data_status["result"] = "success"
                n = ""
            else:
                n = ""
                # data_status["details"]=[]
                # data_status["statuscode"]=455
                # data_status["responseStatus"]=0
                # else:
                # data_status["details"]=[]
                # data_status["statuscode"]=455
                # data_status["responseStatus"]=0
            if n == "not":
                data_status["details"] = []
                data_status["statuscode"] = 455
                data_status["responseStatus"] = 0

            return data_status

@app.route("/jobCompletion",methods=["POST"])
def jobComplectionPage():
    import time
    data_status={"responseStatus":0,"result":""}
    job_id=request.form.get("job_id")
    employee_id=request.form.get("employee_id")
    manager_id=request.form.get("manager_id")
    if request.method=="POST":
        cur = mysql.cursor()
        cur.execute("select id,fk_job_id,fk_employee_id,fk_manager_id,completion from retina_scanner.employee_job_details")
        ids = cur.fetchall()
        for i in ids:
            print(i[1],i[2],i[3])

            n="not"
            if (i[1]==int(job_id)) and (i[2]==int(employee_id)) and (i[3]==int(manager_id)) and (i[4]==0):


                #currenttime_str = time.ctime()
                date=datetime.datetime.now()
            #print(date,i[0])

            # sql="update retina_scanner.employee_job_details set fk_job_id={} where id={}".format(0,i[0])
            #sql="UPDATE retina_scanner.employee_job_details SET completion = 1, completion_timestamp=%s WHERE id=%s",(date,i[0])
                cur.execute("UPDATE retina_scanner.employee_job_details SET completion = 1, completion_timestamp=%s WHERE id=%s",(date,i[0]))
            # print(cur.execute(sql))
            #cursor.execute("update retina_scanner.employee_job_details set completion='"+st where id={}".format(i[0]))
                mysql.commit()
            #print("================")
                data_status["responseStatus"]=1
                data_status["statuscode"]=200
                data_status["result"]="success"
                n=""

            else:
                n=""
                continue

        if n=="not":
            data_status["statuscode"]=455
            data_status["responseStatus"]=0
            data_status["result"]="job with employee does not exist"
    #print(ids)
    return data_status



@app.route("/get_manager_employees",methods=["POST","GET"])
def magaer_user_details():
    data_status={"responseStatus":0,"result":""}
    manager_id=request.form.get("manager_id")
    if request.method=="POST":
        query="select * from retina_scanner.user where fk_manager_id={}".format(manager_id);
        cursor=mysql.cursor()
        cursor.execute(query)
        data=cursor.fetchall()
        if data:
            #data1=list(data)


            employee=[]
            for i in data:
                print(i)
                dic={}
                dic["id"]=i[0]
                dic["first_name"]=i[1]
                dic["last_name"]=i[2]
                dic["phone"]=i[3]
                dic["email"]=i[4]
                dic["aadhar"]=i[10]
                dic["address"]=i[13]
                dic["village"]=i[16]
                dic["district"]=i[15]
                dic["state"]=i[14]
                dic["image"]=i[18]
                dic["left_eye"]=i[19]
                dic["right_eye"]=i[20]
                dic["type"]=i[21]
                dic["fk_manager_id"]=i[22]
                employee.append(dic)
            data_status["responseStatus"]=1
            data_status["result"]="Success"
            data_status["Employee_list"]=employee
            data_status["statusCode"]=200
        else:
            data_status["responseStatus"]=0
            data_status["Employee_list"]=[]
            data_status["result"]="success"
            data_status["statusCode"]=200
        return data_status

@app.route("/get_attendance_records",methods=["POST"])
def getAttendanceRecord():
    data_status={"responseStatus":0,"result":""}
    manager_id = request.form.get("manager_id")
    date=request.form.get("date")
    #print(date)
    if request.method=="POST":
        cur = mysql.cursor()
        cur.execute("SELECT u.id,CONCAT(u.first_name,' ', u.last_name),ui.in_time_attend,ui.out_time_attend,ui.in_time_lat,ui.in_time_long,ui.out_time_lat,ui.out_time_long FROM user u JOIN user_login_info ui WHERE u.id = ui.fk_user_id AND u.fk_manager_id = %s AND  DATE(in_time_attend) = %s",(manager_id,date))
        employee_info=cur.fetchall()
        print(employee_info)
        cur.execute("SELECT COUNT(*) FROM user WHERE fk_manager_id ={}".format(manager_id))
        emp_count=cur.fetchall()
        #print("----------------------------------------------------------------------",emp_count)
        emp_dict={}
        emp_list=[]
        if employee_info:
            print(employee_info)
            for emp in employee_info:
                print(emp)
                #print(type(emp))
                emp_dict={
                "user_id":emp[0],
                "name":emp[1],
                "in_time":emp[2],
                "in_time_lat":emp[4],
                "in_time_long":emp[5],
                "out_time_lat":emp[6],
                "out_time_long":emp[7]
                }
                if emp[3]==None:
                    emp_dict["out_time"]=""
                else:
                     emp_dict["out_time"]=emp[3]
                emp_list.append(emp_dict)
            data_status["responseStatus"]=1
            data_status["result"]="Success"
            data_status["employee_list"]=emp_list
            print(emp_count)
            data_status["employee_count"]=emp_count[0][0]
            data_status["statusCode"]=200
        else:
            data_status["responseStatus"]=1
            data_status["result"]="success"
            data_status["employee_list"]=[]
            data_status["employee_count"]=emp_count[0][0]
            data_status["statusCode"]=200
    return data_status

@app.route("/get_job_information",methods=["POST","GET"])
def get_job_information():
    data_status={"responseStatus":0,"result":""}
    state1=request.form.get("state")
    district=request.form.get("district")
    if request.method=="POST":
        #query="select * from retina_scanner.job_details where state='{}'".format(state1);
        cursor=mysql.cursor()
        cursor.execute("select * from retina_scanner.job_details where state=%s and district=%s",(state1,district))
        data=cursor.fetchall()
        if data:
            #data1=list(data)

            print(data)
            employee=[]
            for i in data:
                print(i)
                dic={}
                dic["job_id"]=i[0]
                dic["job_name"]=i[1]
                dic["job_description"]=i[2]
                dic["state"]=i[3]
                dic["district"]=i[4]
                dic["block"]=i[5]
                dic["panchayat"]=i[6]
                dic["sector"]=i[7]
                dic["job_duration_in_days"]=i[8]
                dic["number_of_people_needed"]=i[9]
                #dic["image"]=i[10]
                #dic["left_eye"]=i[11]
                #dic["right_eye"]=i[12]
                employee.append(dic)
            data_status["responseStatus"]=1
            data_status["result"]="Success"
            data_status["Employee_list"]=employee
            data_status["statusCode"]=200
        else:
            data_status["responseStatus"]=0
            data_status["Employee_list"]=[]
            data_status["result"]="success"
            data_status["statusCode"]=200
        return data_status


@app.route("/set_job_employee",methods=["POST","GET"])
def set_job_employee():
    data_status={"responseStatus":0,"result":""}
    job_id=request.form.get("job_id")
    employee_id=request.form.get("employee_id")
    manager_id=request.form.get("manager_id")
    start_date=request.form.get("start_date")
    end_date=request.form.get("end_date")
    date=datetime.datetime.now()


    if request.method=="POST":

        cursor=mysql.cursor()
        cursor.execute("insert into retina_scanner.employee_job_details values(%s,%s,%s,%s,%s,%s,%s,%s,%s)",(None,job_id,employee_id,manager_id,start_date,end_date,0,None,date))
        data=cursor.fetchall()
        mysql.commit()
        data_status["responseStatus"]=1
        data_status["result"]="success"
        data_status["statusCode"]=200
        return data_status

@app.route('/attendanceOffline', methods=['POST', 'GET'])
def attendanceOffline():
    mycursor = mysql.cursor()
    data_status = {"responseStatus": 0, "result": ""}
    if request.method == 'POST':
        data=request.get_json()
        for i in data['employeeList']:
            #print(data['employeeList'][0]['employe_id'])
            print(i)
            manager_id=data['manager_id']
            employee_id=i['employee_id']
            added_timestamp=datetime.datetime.now()
            mark_lat=i['lat']
            mark_long = i['long']
            attend_type =i['type(in/out)']
            mark_time=i['mark_time']
            file1=i['file']



            mycursor.execute("insert into retina_scanner.offline_attendance values(%s,%s,%s,%s,%s,%s,%s,%s,%s)", (None,manager_id,employee_id,mark_time,mark_lat,mark_long,attend_type,file1,added_timestamp))
            mysql.commit()
        mycursor.close()
        data_status["responseStatus"] = 1
        data_status["result"] = "success"
        data_status["statusCode"]=200
    return data_status




if __name__=="__main__":
    app.run(debug=True)



