from flask import Flask, render_template,request,session
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
import sqlite3
import xgboost as xgb
from flask import render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from attention import attention #loading attention transformer layer
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, RepeatVector, Input, Conv2D, UpSampling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
import copy
import itertools
import os

app = Flask(__name__)

#defining sign labels available in dataset
labels = ['I', 'apple', 'can', 'get', 'good', 'have', 'help', 'how', 'like', 'love', 'my', 'no', 'sorry', 'thank-you', 'want', 'yes', 'you', 'your']

#defining open pose parameters threshold to detect hand keypoints 
min_detection_confidence = 0.4
min_tracking_confidence = 0.4
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
drawingModule = mp.solutions.drawing_utils
min_detection_confidence = 0.4
min_tracking_confidence = 0.4
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

global X, Y, words, signs, sign_label, vectorizer, X_train, X_test, y_train, y_test, sc
global words_X_train, words_X_test, words_y_train, words_y_test, encoder_model, decoder_model
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/adminlogin')
def AdminLogin():
    return render_template('AdminApp/AdminLogin.html')

@app.route('/AdminAction', methods=['POST'])
def AdminAction():
    if request.method == 'POST':
        username=request.form['username']
        password=request.form['password']

        if username=='Admin' and password=='Admin':
            return render_template("AdminApp/AdminHome.html")
        else:
            context={'msg':'Login Failed..!!'}
            return render_template("AdminApp/AdminLogin.html",**context)

@app.route('/AdminHome')
def AdminHome():
    return render_template("AdminApp/AdminHome.html")

@app.route('/Upload')
def Upload():
    return render_template("AdminApp/Upload.html")


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global dataset,filepath
global X, Y, words, signs, sign_label, vectorizer, X_train, X_test, y_train, y_test, sc
global words_X_train, words_X_test, words_y_train, words_y_test, encoder_model, decoder_model

@app.route('/uploadDataset')
def uploadDataset():
    global X, Y, words, signs, sign_label, vectorizer, labels
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
        words = np.load("model/words.npy")
        signs = np.load("model/signs.npy")
        sign_label = np.load("model/sign_label.npy")
    else:
        words = []
        signs = []
        sign_label = []
        old = "old"
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                if 'Thumbs.db' not in directory[j]:
                    current = os.path.basename(root)
                    if current != old:
                        img = cv2.imread(root+"/"+directory[j],0)
                        img = cv2.resize(img, (128, 128))
                        sign_label.append(img)
                        old = current
                    name = int(os.path.basename(root))
                    data = labels[name]
                    data = data.strip().lower()
                    words.append(data)
                    signs.append(img)
        words = np.asarray(words)
        signs = np.asarray(signs)
        np.save("model/words", words)
        np.save("model/signs", signs)
    text.insert(END,"Total images found in Dataset : "+str(signs.shape[0])+"\n\n")    
    text.insert(END,"Different signs words found in Dataset = "+str(labels))
    #visualizing class labels count found in dataset
    names, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (12, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Signs Class Label Graph")
    plt.ylabel("Count")
    plt.show()
@app.route('/processDataset')
def processDataset():
    global X, Y, words, signs, sign_label, vectorizer, sc
    text.delete('1.0', END)
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=39)
    words = vectorizer.fit_transform(words).toarray()
    XX = []
    for i in range(len(words)):
        word = words[i]
        word = np.reshape(word, (13, 3))
        word = cv2.resize(word, (128, 128))
        XX.append(word)
    words = np.asarray(XX)    
    words = np.reshape(words, (words.shape[0], words.shape[1], words.shape[2], 1))
    signs = np.reshape(signs, (signs.shape[0], signs.shape[1], signs.shape[2], 1))
    words = words.astype('float32')
    words = words/255
    signs = signs.astype('float32')
    signs = signs/255
    sc = StandardScaler()
    X = sc.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    text.insert(END,"Images shuffling, word vectorization & Normalization Completed")
@app.route('/splitDataset')
def splitDataset():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y, words, signs
    global words_X_train, words_X_test, words_y_train, words_y_test
    words_X_train, words_X_test, words_y_train, words_y_test = train_test_split(words, signs, test_size = 0.2, random_state = 1)#splitting dataset into train test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"80% Signs used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% Signs used for testing : "+str(X_test.shape[0])+"\n\n")
@app.route('/trainSignnet')
def trainSignnet():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y, words, signs
    global words_X_train, words_X_test, words_y_train, words_y_test, labels, encoder_model, decoder_model
    #training encoder from sign to text
    encoder_model = Sequential()
    #defining cnn layer
    encoder_model.add(Convolution2D(64, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    encoder_model.add(MaxPooling2D(pool_size = (1, 1)))
    encoder_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    encoder_model.add(MaxPooling2D(pool_size = (1, 1)))
    encoder_model.add(Flatten())
    encoder_model.add(RepeatVector(3))
    encoder_model.add(attention(return_sequences=True,name='attention')) # ========define transformer Attention layer
    encoder_model.add(LSTM(32))
    encoder_model.add(Dense(units = 64, activation = 'relu'))
    encoder_model.add(Dense(units = y_train.shape[1], activation='softmax'))
    #compiling, training and loading model
    encoder_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    encoder_model.load_weights("model/encoder_weights.keras")
    input_img = Input(shape=(128, 128, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder_model = Model(input_img, decoded)
    decoder_model.compile(optimizer='adam', loss='binary_crossentropy')
    if os.path.exists("model/decoder_weights.keras") == False:
        model_check_point = ModelCheckpoint(filepath='model/decoder_weights.keras', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(words_X_train, words_y_train, batch_size = 32, epochs = 380, validation_data=(words_X_test, words_y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/decoder_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        decoder_model.load_weights("model/decoder_weights.keras")
    predict = encoder_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict_sent = ""
    true_sent = ""
    for i in range(len(predict)):
        true_sent += labels[y_test1[i]]+" "
        predict_sent += labels[predict[i]]+" "
    true_sent = true_sent.strip()
    predict_sent = predict_sent.strip()
    score = sentence_bleu([true_sent.split()], predict_sent.split()) / 2
    text.insert(END,"SignNetII Bleu Score = "+str(score))
@app.route('/getSignImage')
def getSignImage(word, index):
    print(index)
    test_data = word
    test_data = test_data.lower().strip()
    test_data = vectorizer.transform([test_data]).toarray()
    test_data = np.reshape(test_data, (13, 3))
    test_data = cv2.resize(test_data, (128, 128))
    temp = []
    temp.append(test_data)
    temp = np.asarray(temp)    
    temp = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2], 1))
    temp = temp.astype('float32')
    temp = temp/255
    predict = decoder_model.predict(temp)
    predict = sign_label[index]
    predict = cv2.resize(predict, (300, 300))
    return predict
@app.route('/predict')
def predict():
    global encoder_model, decoder_model
    text.delete('1.0', END)
    text.insert(END,"SignNet Output\n\n")
    global sc, min_detection_confidence, min_tracking_confidence, labels
    filename = askopenfilename(initialdir = "testVideo")
    camera = cv2.VideoCapture(filename)
    detected = 0
    while(True):
        (grabbed, frame) = camera.read()
        if frame is not None:
            img = cv2.flip(frame, 1)
            debug_image = copy.deepcopy(img)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    data = []
                    data.append(pre_processed_landmark_list)
                    data = np.asarray(data)
                    data = sc.transform(data)
                    data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
                    predict = encoder_model.predict(data)
                    predicted_text = np.argmax(predict)
                    cv2.putText(img, 'Sign to Text : '+labels[predicted_text], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
                    text.insert(END,labels[predicted_text]+"\n")
                    text.update_idletasks()
                    sign_image = getSignImage(labels[predicted_text], predicted_text)
                    detected = 1                                       
            cv2.imshow("Sign Langauge Prediction", img)
            if detected == 1:
                cv2.imshow("Text to Sign", sign_image)
                detected = 0
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break            
        else:
            break
    camera.release()
    cv2.destroyAllWindows() 
    
@app.route('/predictfromWebcam')
def predictfromWebcam():
    global sign_label, vectorizer, sc
    global encoder_model, decoder_model
    text.delete('1.0', END)
    text.insert(END,"SignNet Output\n\n")
    global model, sc, min_detection_confidence, min_tracking_confidence, labels
    camera = cv2.VideoCapture(0)
    count = 0
    while(True):
        (grabbed, frame) = camera.read()
        if frame is not None:
            #frame = cv2.resize(frame, (1000, 800))
            img = cv2.flip(frame, 1)
            debug_image = copy.deepcopy(img)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    data = []
                    data.append(pre_processed_landmark_list)
                    data = np.asarray(data)
                    data = sc.transform(data)
                    data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
                    predict = encoder_model.predict(data)
                    predicted_text = np.argmax(predict)
                    cv2.putText(img, 'Sign to Text : '+labels[predicted_text], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
                    text.insert(END,labels[predicted_text]+"\n")
                    text.update_idletasks()
                    sign_image = getSignImage(labels[predicted_text], predicted_text)
                    cv2.imshow("Text to Sign", sign_image)
                    count += 1
            cv2.imshow("Sign Langauge Prediction", img)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
            if count > 1500:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows() 
@app.route('/userlogin')
def userlogin():
    return render_template('UserApp/Login.html')

@app.route('/register')
def register():
    return render_template('UserApp/Register.html')
@app.route('/RegAction', methods=['POST'])
def RegAction():
    if request.method == 'POST':
        name=request.form['name']
        email=request.form['email']
        mobile=request.form['mobile']
        username=request.form['username']
        password=request.form['password']

        con=sqlite3.connect('database.db')
        cur=con.cursor()
        #cur.execute("create table user(name varchar(100),email varchar(200),mobile varchar(200),username varchar(100),password varchar(100))")
        cur.execute("select * from user where username='"+username+"'and password='"+password+"'")
        data=cur.fetchone()
        if data is None:
            cur=con.cursor()
            cur.execute("insert into user values('"+name+"','"+email+"','"+mobile+"','"+username+"','"+password+"')")
            con.commit()
            return render_template('UserApp/Register.html', msg="Successfully Registered..!!")
        else:
            return render_template('UserApp/Register.html', msg="username and password is already exist..!!")

app.secret_key = '123'
@app.route('/UserAction', methods=['POST'])
def UserAction():
    username=request.form['username']
    password=request.form['password']

    con=sqlite3.connect('database.db')
    cur=con.cursor()
    cur.execute("select * from user where username='"+username+"'and password='"+password+"'")
    data=cur.fetchone()
    if data is None:
        return render_template('UserApp/Login.html', msg="Login Failed..!!")
    else:
        session['username'] =data[3]
        return render_template('UserApp/Home.html',username=session['username'])


@app.route('/UserHome')
def UserHome():
    return render_template('UserApp/Home.html')


if __name__ == '__main__':
    app.run(debug=True)


