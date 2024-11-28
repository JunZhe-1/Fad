# Flask-related imports #
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, send_file, session
from wtforms import Form, StringField, RadioField, SelectField, TextAreaField, validators, ValidationError, PasswordField
import sys
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import shelve, re
from flask_wtf import FlaskForm
from Forms import configurationForm, emailForm, LoginForm, RegisterForm, MFAForm, FeedbackForm
from flask_mail import Mail, Message
import random

# Object detection & processing-related imports #
import torch
import torchvision
import cv2
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Datetime-related imports #
import datetime, time
import threading
from datetime import datetime, timedelta
from flask_apscheduler import APScheduler
import numpy as np
import pytz

# File-reader-related imports #
import openpyxl
from openpyxl.styles import Font

# OOP-related imports #
from OOP import *
import os

# Back-end codes for object detection & processing #
if torch.cuda.is_available():
    print('you are using gpu to process the video camera')
else:
    print('no gpu is found in this python environment. using cpu to process')

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        self.condition = threading.Condition()
        self.is_running = False
        self.frame = None
        self.pellets_num = 0
        self.callback = None
        super().__init__(name=name)
        self.start()

    def start(self):
        self.is_running = True
        super().start()

    def stop(self, timeout=None):
        self.is_running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.is_running:
            (rv, img) = self.capture.read()
            assert rv
            counter += 1
            with self.condition:
                self.frame = img if rv else None
                self.condition.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, sequence_number=None, timeout=None):
        with self.condition:
            if wait:
                # If sequence_number is not provided, get the next sequence number
                if sequence_number is None:
                    sequence_number = self.pellets_num + 1

                if sequence_number < 1:
                    sequence_number = 1

                if (sequence_number) > 0:
                    self.pellets_num = sequence_number

                # Wait until the latest frame's sequence number is greater than or equal to sequence_number
                rv = self.condition.wait_for(lambda: self.pellets_num >= sequence_number, timeout=timeout) # if there is a pellets. should get "true"
                if not rv:
                    return (self.pellets_num, self.frame)  # Return the latest frame if timeout occurs
            return (self.pellets_num, self.frame)  # Return the latest frame

# define the id "1" for pellets
# do note that in the pth file, the pellet id also is 1
class_labels = {
    1: 'Pellets',
    2: 'Fecal Matters'
}

# pth file where you have defined on roboflow
model_path = './best_model.pth'
latest_processed_frame = None  # Stores the latest processed frame
stop_event = threading.Event()  # Event to stop threads gracefully
freshest_frame = None

# Initialize variables for feeding logic
feeding = False
feeding_timer = None
showing_timer = None
line_chart_timer = None
object_count = {1: 0}
frame_data = {
    'object_count': {1: 0},  # Initialize with default values for object count
    'bounding_boxes': []  # List to store bounding boxes for the current frame
}

# Initialize locks for shared variables
latest_processed_frame_lock = threading.Lock()
feeding_lock = threading.Lock()
frame_data_lock = threading.Lock()
object_count_lock = threading.Lock()
freshest_frame_lock = threading.Lock()

def create_model(num_classes, pretrained=False, coco_model=False):
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    if not coco_model:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Function to load the custom-trained model from the .pth file
def load_model(model_path, num_classes):
    model = create_model(num_classes=num_classes, pretrained=False, coco_model=False)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model




# Assume these methods to load model and settings are defined
model = load_model(model_path, num_classes=2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

def check_feeding_time():
    global feeding, feeding_timer, showing_timer, line_chart_timer
    db = shelve.open('settings.db', 'w')
    Time_Record_dict = db['Time_Record']
    db.close()

    setting = Time_Record_dict.get('Time_Record_Info')

    hours, minutes = setting.get_first_timer().split(':')
    hours1, minutes1 = setting.get_second_timer().split(':')

    first_feeding_time = int(hours)
    first_feeding_time_min = int(minutes)
    second_feeding_time = int(hours1)
    second_feeding_time_min = int(minutes1)
    while not stop_event.is_set():
        current_time = datetime.now()
        # Check if it's time for feeding (at the exact second)
        if (current_time.hour == first_feeding_time or current_time.hour == second_feeding_time) and \
           (current_time.minute == first_feeding_time_min or current_time.minute == second_feeding_time_min) and \
           current_time.second == 0:
            feeding = True
            feeding_timer = None
            showing_timer = None
            line_chart_timer = time.time()
        else:
            feeding = False  # If it's not time for feeding, set feeding to False
        time.sleep(1)  # Check every second

def capture_frames():
    cap = cv2.VideoCapture('rtsp://admin:fyp2024Fish34535@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0')
    global  freshest_frame
    with freshest_frame_lock:
        freshest_frame = FreshestFrame(cap)
    global latest_processed_frame
    while not stop_event.is_set():
        # Wait for the newest frame
        sequence_num, frame = freshest_frame.read(wait=True)
        if frame is not None:
            # Process the frame
            with latest_processed_frame_lock:
                latest_processed_frame = frame
        time.sleep(0.03)  # Adjust to control frame rate (~30 FPS)



def process_frames():
    # define the dictionary to store the number of pellets
    # Assuming 1 class for 'Pellet'
    global freshest_frame
    global frame_data
    global object_count
    object_count = {1: 0}
    bounding_boxes = []
    global latest_processed_frame
    global feeding
    global feeding_timer
    db = shelve.open('settings.db', 'w')
    Time_Record_dict = db['Time_Record']
    db.close()

    setting = Time_Record_dict.get('Time_Record_Info')

    hours, minutes = setting.get_first_timer().split(':')
    hours1, minutes1 = setting.get_second_timer().split(':')

    first_feeding_time = int(hours)
    first_feeding_time_min = int(minutes)
    second_feeding_time = int(hours1)
    second_feeding_time_min = int(minutes1)

    confidence = float(setting.get_confidence())/100

    showing_timer = None
    line_chart_timer, email_TF = (None,False)
    desired_time = None

    formatted_desired_time = None
    current_datetime = datetime.now()
    frame_counter = 0  # Counter to track frames
    while True:
        bounding_boxes = []
        # Process the predictions and update object count
        temp_object_count = {1: 0}  # Initialize count for the current frame

        frame_counter += 1
        if frame_counter % 3 != 0:  # Skip 2 out of every 3 frames
            time.sleep(0.03)
            continue
        current_time = datetime.now().time()
        if (current_time.hour == first_feeding_time or current_time.hour == second_feeding_time) and (current_time.minute == first_feeding_time_min or current_time.minute == second_feeding_time_min) and current_time.second == 0:
            with feeding_lock:
                feeding = True
            feeding_timer = None
            showing_timer = None
            line_chart_timer = time.time()
        with freshest_frame_lock:
            if freshest_frame is not None:
                cnt, frame = freshest_frame.read(sequence_number=object_count[1] + 1)
                if frame is None:
                    break

            # Preprocess the frame
            img_tensor = torchvision.transforms.ToTensor()(frame).to(device)
            img_tensor = img_tensor.unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                predictions = model(img_tensor)

            for i in range(len(predictions[0]['labels'])):
                label = predictions[0]['labels'][i].item()

                if label in class_labels:
                    box = predictions[0]['boxes'][i].cpu().numpy().astype(int) # used to define the size of the object
                    score = predictions[0]['scores'][i].item() #the probability of the object

                    if (label == 1 and score > confidence):
                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) #(0,255,0) is the color (blue, green, yellow)
                        cv2.putText(frame, f'{class_labels[label]}: {score:.2f}', (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        temp_object_count[label] += 1
                        bounding_boxes.append((box, label, score))

                        # Start feeding timer if pellets are detected
                        if label == 1 and feeding_timer is None and feeding:
                            feeding_timer = time.time()

            # store the pellets number to the object count which is permanently
            for label, count in temp_object_count.items():
                if label == 1:  # Assuming label 1 represents 'Pellets'
                    with object_count_lock:
                        object_count[label] = count

            # Check feeding timer and switch to stop feeding if required
            if feeding_timer is not None and feeding:
                elapsed_time = (time.time() - feeding_timer)
                print( f'elapsed time: {elapsed_time:.3f}' )
                with object_count_lock:
                    if elapsed_time > int(setting.get_seconds()) and sum(object_count.values()) > int(setting.get_pellets()):
                        with feeding_lock:
                            feeding = False
                        feeding_timer = None
                        showing_timer = time.time()

                    # change to None when there is no pellets
                    elif object_count[1] == 0:
                        feeding_timer = None
            with object_count_lock:
            # Display the frame with detections and object count
                for label, count in object_count.items():
                    text = f'{class_labels[label]} Count: {count}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    text_position = (frame.shape[1] - text_size[0] - 10, 30 * (label+1))
                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 255, 255), 2)
            with object_count_lock:
                # Display feeding or stop feeding text just below the object counter
                text_position_feed = (frame.shape[1] - text_size[0] - 10  , 30 * (max(object_count.keys()) + 1))

            if feeding:
                cv2.putText(frame, "Feeding...", text_position_feed,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            else:
                if showing_timer is not None:
                    i = time.time() - showing_timer

                    if i > 3:
                        showing_timer = None
                        j = time.time() - line_chart_timer

                        line_chart_timer = None

                        db = shelve.open('line_chart_data.db', 'w')
                        Line_Chart_Data_dict = db['Line_Chart_Data']

                        current_date = datetime.today().strftime("%Y-%m-%d")

                        if current_date in Line_Chart_Data_dict:
                            Line_chart_objects = Line_Chart_Data_dict.get(current_date)
                            Line_chart_objects.set_timeRecord(j + Line_chart_objects.get_timeRecord())
                            Line_Chart_Data_dict[current_date] = Line_chart_objects
                        elif current_date not in Line_Chart_Data_dict:
                            print(current_time," is not in the dictionary. creating a new one...")
                            new_object = Line_Chart_Data(current_date, j)
                            Line_Chart_Data_dict[current_date] = new_object
                        db['Line_Chart_Data'] = Line_Chart_Data_dict
                        db.close()

                        if (current_time.hour >= first_feeding_time) and (current_time.hour >=second_feeding_time and current_time.minute >second_feeding_time_min):
                            print('sending email feature')


                        for today_date in Line_Chart_Data_dict:
                            Line_chart_objects = Line_Chart_Data_dict.get(today_date)
                            print(Line_chart_objects.get_date(),': ', Line_chart_objects.get_timeRecord())

                        print('running in website')
                    else:
                        cv2.putText(frame, "Stop Feeding", text_position_feed,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                else:
                    if (current_time.hour <= first_feeding_time and current_time.minute <= first_feeding_time_min) or current_time.hour < first_feeding_time:
                        desired_time = current_datetime.replace(hour=first_feeding_time, minute=first_feeding_time_min, second=0,
                                                                    microsecond=0)
                        formatted_desired_time = 'Next Round: '+ desired_time.strftime("%I:%M %p")

                        text_size = cv2.getTextSize(formatted_desired_time, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                        text_position = (frame.shape[1] - text_size[0] - 10, 30 * 4)
                        cv2.putText(frame, formatted_desired_time, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (0, 255, 0), 2)



                    elif ((current_time.hour <= second_feeding_time and current_time.minute <= second_feeding_time_min)) or (current_time.hour < second_feeding_time):
                        desired_time = current_datetime.replace(hour=second_feeding_time, minute=second_feeding_time_min, second=0,
                                                                microsecond=0)
                        formatted_desired_time = 'next round: '+ desired_time.strftime("%I:%M %p")

                        text_size = cv2.getTextSize(formatted_desired_time, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                        text_position = (frame.shape[1] - text_size[0] - 10, 30 * 4)
                        cv2.putText(frame, formatted_desired_time, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                    else:
                        # Add one day to the current date and time
                        next_day = current_datetime + timedelta(days=1)
                        # Set desired_time to 8 AM of the next day
                        desired_time = next_day.replace(hour=first_feeding_time, minute=first_feeding_time_min, second=0, microsecond=0)

                        formatted_desired_time = 'Tomorrow at: ' +desired_time.strftime("%I:%M %p")

                        text_size = cv2.getTextSize(formatted_desired_time, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                        text_position = (frame.shape[1] - text_size[0] - 10, 30 * 4)
                        cv2.putText(frame, formatted_desired_time, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (0, 0, 255), 2)
            with frame_data_lock:
                frame_data['object_count'] = temp_object_count
                frame_data['bounding_boxes'] = bounding_boxes
            with latest_processed_frame_lock:
                latest_processed_frame = frame

def generate_frames():
    global latest_processed_frame, frame_data
    while not stop_event.is_set():
        with latest_processed_frame_lock:
            frame = latest_processed_frame.copy()  # Create a copy of the frame to avoid modification of original
        if frame is not None:
            # Display object count and bounding boxes from frame_data
            with frame_data_lock:
                for label, count in frame_data['object_count'].items():
                    text = f'{class_labels[label]} Count: {count}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    text_position = (frame.shape[1] - text_size[0] - 10, 30 * (label + 1))
                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 255, 255), 2)

                # Draw bounding boxes from frame_data
                for box, label, score in frame_data['bounding_boxes']:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_labels[label]}: {score:.2f}', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert frame to jpeg and yield it
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        time.sleep(0.03)  # Adjust the frame rate if necessary


# Web application #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Initialize Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = '2931801324qq@gmail.com'
app.config['MAIL_PASSWORD'] = 'qqmr eawg svut gysf'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


# Dictionaries #
#j = datetime.datetime.now()
# print(j)
Time_Record_dict = {}
Line_Chart_Data_dict = {}
Email_dict = {}

# User model
class User(UserMixin):
    def __init__(self, username, email, password):
        self.id = username  # Use username as ID for simplicity
        self.username = username
        self.email = email
        self.password = password

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    with shelve.open('users.db') as db:
        user_data = db.get(user_id)
        if user_data:
            return User(user_data['username'], user_data['email'], user_data['password'])
        else:
            return None


# Default route to redirect to login page
@app.route('/')
@login_required
def index():
    return redirect(url_for('logout'))

# Function to open shelve safely
def open_shelve(filename, mode='c'):
    try:
        shelf = shelve.open(filename, mode)
        return shelf
    except Exception as e:
        print(f"Error opening shelve: {e}")
        return None

# Routes for Registration and Login
# Routes for Registration and Login using shelve
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()  # Create an instance of RegisterForm

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        confirm_password = form.confirm_password.data

        # Check if the username or email already exists in the database
        with shelve.open('users.db', 'c') as db:
            username_exists = username in db
            email_exists = any(user_data['email'] == email for user_data in db.values())

            if username_exists:
                flash('Username or email already in use', 'danger')
            elif email_exists:
                flash('Username or email already in use', 'danger')
            else:
                # If neither username nor email is already registered, proceed with registration
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username, email, hashed_password)
                db[username] = {'username': new_user.username, 'email': new_user.email, 'password': new_user.password}
                flash('You are now registered and can log in', 'success')
                return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        try:
            with shelve.open('users.db', 'c') as db:
                if username in db:
                    stored_password_hash = db[username]['password']
                    if check_password_hash(stored_password_hash, password):
                        # Passwords match, proceed with MFA
                        user_email = db[username]['email']

                        # Generate a 6-digit MFA code
                        mfa_code = str(random.randint(100000, 999999))
                        session['email'] = user_email  # Save email in session
                        session['mfa_code'] = mfa_code  # Store in session
                        session['username'] = username  # Save username for next steps

                        # Send the code via email
                        msg = Message('MFA Code',
                                        sender='2931801324qq@gmail.com',
                                        recipients=[user_email])
                        msg.body = f'Your 6-digit MFA code is {mfa_code}'
                        try:
                            print(mfa_code)
                            #mail.send(msg)
                            flash('An authentication code has been sent to your email.', 'info')
                        except:
                            flash('Error sending MFA', 'danger')

                        return redirect(url_for('mfa_verify'))  # Redirect to MFA verification page
                    else:
                        flash('Invalid login credentials', 'danger')
                else:
                    flash('Invalid login credentials', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')

    return render_template('login.html', form=form)


@app.route('/mfa-verify', methods=['GET', 'POST'])
def mfa_verify():
    form = MFAForm()

    if form.validate_on_submit():
        entered_code = form.code.data

        if entered_code == session.get('mfa_code'):
            # MFA passed, log the user in
            username = session.get('username')

            # Use context manager to ensure the shelf is properly opened and closed
            with shelve.open('users.db', 'r') as db:
                user_email = db[username]['email']
                hashed_password = db[username]['password']

            user = User(username, user_email, hashed_password)
            login_user(user)
            flash('You are now logged in', 'success')
            session.pop('mfa_code')  # Clear MFA code after success
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid authentication code', 'danger')

    return render_template('mfa_verify.html', form=form)

@app.route('/logout')
def logout():
    session.clear()
    flash('Please log in to access.', 'info')
    return redirect(url_for('login'))


from flask import Flask, flash, render_template, request, redirect, session, url_for
import shelve, re

# Helper function to get settings from the database
def get_settings():
    with shelve.open('settings.db', 'c') as db:
        settings = db.get('settings', {'interval': 2000, 'threshold': 5})  # Default interval is 2000ms and default threshold is 5
        app.logger.debug(f"Settings read from database: {settings}")
    return settings

def update_settings(new_settings):
    with shelve.open('settings.db', 'c') as db:
        db['settings'] = new_settings
        app.logger.debug(f"Settings updated in database: {new_settings}")

# Route to update the interval setting
@app.route('/update_interval', methods=['POST'])
def update_interval():
    try:
        interval = request.json.get('interval')
        app.logger.debug(f"Received interval: {interval}")  # Debug log

        # Validate interval (should be a positive integer)
        if interval is not None:
            interval = int(interval)
            if interval <= 0:
                raise ValueError("Interval should be a positive integer.")

            # Update interval in settings
            settings = get_settings()
            settings['interval'] = interval
            update_settings(settings)
            app.logger.debug(f"Interval updated successfully to {interval}")
            return jsonify({'message': 'Interval updated successfully'}), 200
        else:
            return jsonify({'error': 'Interval value not provided'}), 400

    except ValueError as e:
        app.logger.error(f"Invalid interval value: {e}")
        return jsonify({'error': 'Invalid interval value. Must be a positive integer.'}), 400

    except Exception as e:
        app.logger.error(f"An error occurred while updating interval: {str(e)}")
        return jsonify({'error': 'An error occurred while updating interval.'}), 500

# Route to retrieve the current interval setting
@app.route('/get_interval', methods=['GET'])
def get_interval():
    try:
        settings = get_settings()
        app.logger.debug(f"Current interval retrieved: {settings['interval']}")
        return jsonify({'interval': settings['interval']}), 200
    except Exception as e:
        app.logger.error(f"An error occurred while retrieving interval: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving interval.'}), 500

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    try:
        threshold = request.json.get('threshold')
        app.logger.debug(f"Received threshold: {threshold}")  # Debug log

        # Validate threshold (should be a positive integer)
        if threshold is not None:
            threshold = int(threshold)
            if threshold <= 0:
                raise ValueError("Threshold should be a positive integer.")

            # Update threshold in settings
            settings = get_settings()
            settings['threshold'] = threshold
            update_settings(settings)
            app.logger.debug(f"Threshold updated successfully to {threshold}")
            return jsonify({'message': 'Threshold updated successfully'}), 200
        else:
            return jsonify({'error': 'Threshold value not provided'}), 400

    except ValueError as e:
        app.logger.error(f"Invalid threshold value: {e}")
        return jsonify({'error': 'Invalid threshold value. Must be a positive integer.'}), 400

    except Exception as e:
        app.logger.error(f"An error occurred while updating threshold: {str(e)}")
        return jsonify({'error': 'An error occurred while updating threshold.'}), 500

@app.route('/get_threshold', methods=['GET'])
def get_threshold():
    try:
        settings = get_settings()
        app.logger.debug(f"Current threshold retrieved: {settings['threshold']}")
        return jsonify({'threshold': settings['threshold']}), 200
    except Exception as e:
        app.logger.error(f"An error occurred while retrieving threshold: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving threshold.'}), 500


@app.route('/pellet_data')
def get_pellet_data():
    # Define test data
    pellet_data = {
        '26 Oct 2024': {'8:05 AM': 35, '6:05 PM': 42},
        '27 Oct 2024': {'8:05 AM': 30, '6:05 PM': 40},
        '28 Oct 2024': {'8:05 AM': 28, '6:05 PM': 45},
        '14 Nov 2024': {'8:05 AM': 33, '6:05 PM': 38},
        '15 Nov 2024': {'8:05 AM': 27, '6:05 PM': 41},
        '16 Nov 2024': {'8:05 AM': 29, '6:05 PM': 39},
        '17 Nov 2024': {'8:05 AM': 32, '6:05 PM': 37},
        '18 Nov 2024': {'8:05 AM': 31, '6:05 PM': 40},
        '19 Nov 2024': {'8:05 AM': 30, '6:05 PM': 39},
        '20 Nov 2024': {'8:05 AM': 28, '6:05 PM': 36},
        '21 Nov 2024': {'8:05 AM': 33, '6:05 PM': 42},
        '22 Nov 2024': {'8:05 AM': 29, '6:05 PM': 41},
        '23 Nov 2024': {'8:05 AM': 30, '6:05 PM': 38},
        '24 Nov 2024': {'8:05 AM': 32, '6:05 PM': 35}
    }

    # Check if the database exists, and if not, create and populate it
    db_path = 'mock_chart_data.db'
    if not os.path.exists(db_path + '.db'):  # Shelve creates additional files with ".db"
        with shelve.open(db_path, 'c') as db:
            for date, feeds in pellet_data.items():
                db[datetime.strptime(date, "%d %b %Y").strftime("%Y-%m-%d")] = feeds
            db.sync()
        print("Database created and initialized.")

    # Generate the last 7 days (including today)
    current_day = datetime.today().date()
    last_7_days = [(current_day - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]

    # Initialize lists to hold pellet counts
    first_feed_counts = []
    second_feed_counts = []

    # Open the shelve database and retrieve data
    with shelve.open(db_path, 'r') as db:
        for day in last_7_days:
            if day in db:
                first_feed_counts.append(db[day].get('8:05 AM', 0))
                second_feed_counts.append(db[day].get('6:05 PM', 0))
            else:
                first_feed_counts.append(0)
                second_feed_counts.append(0)

    response_data = {
        'labels': [datetime.strptime(day, "%Y-%m-%d").strftime("%d %b") for day in last_7_days],
        'first_feed': first_feed_counts,
        'second_feed': second_feed_counts,
    }

    return jsonify(response_data)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    edit_form = configurationForm(request.form)
    db = shelve.open('settings.db', 'r')
    Time_Record_dict = db['Time_Record']
    db.close()
    id_array = []
    for key in Time_Record_dict:
        product = Time_Record_dict.get(key)
        if key == "Time_Record_Info":
            id_array.append(product)
    # Fetch Pellet Data (You can directly use `get_pellet_data` or emulate its behavior)
    response = get_pellet_data()
    pellet_data = response.get_json()  # Convert the Flask Response to JSON

    return render_template('dashboard.html', count=len(id_array), id_array=id_array, edit=0, form=edit_form,
                           pellet_labels=pellet_data['labels'],first_feed=pellet_data['first_feed'],second_feed=pellet_data['second_feed'])

@app.route('/camera_view',methods=['GET','POST'])
def camera_view():
    return render_template('camera_view.html')

@app.route('/export_data', methods=['POST'])
def export_data():
    # Get data from the request
    data = request.get_json()
    labels = data.get('labels', [])
    values = data.get('data', [])

    # Create an Excel workbook and worksheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = ' Leftover Pellets Over The Past Seven Days'

    # Set up the header
    sheet["A1"] = "Date"
    sheet["B1"] = "Number of pellets"
    sheet["A1"].font = Font(bold=True)
    sheet["B1"].font = Font(bold=True)

    # Populate the worksheet with data
    for index, (label, value) in enumerate(zip(labels, values), start=2):
        sheet[f"A{index}"] = label
        sheet[f"B{index}"] = value

    # Save the workbook to a file
    file_path = 'consumption_data.xlsx'
    workbook.save(file_path)

    return send_file(file_path, as_attachment=True, download_name='leftover_feed_data.xlsx')

@app.route('/update', methods=['GET', 'POST'])
@login_required
def update_setting():
    setting = configurationForm(request.form)

    if request.method == 'POST' and setting.validate():
        pattern = r'^([01]\d|2[0-3]):([0-5]\d)$'

        if re.match(pattern, setting.first_timer.data) and re.match(pattern, setting.second_timer.data):
            first_hour = int(setting.first_timer.data.split(':')[0])
            second_hour = int(setting.second_timer.data.split(':')[0])
            if (6 <= first_hour <=12) and (12<= second_hour <=24):
                db = shelve.open('settings.db', 'w')
                Time_Record_dict = db['Time_Record']

                j = Time_Record_dict.get('Time_Record_Info')
                j.set_first_timer(setting.first_timer.data)
                j.set_second_timer(setting.second_timer.data)
                j.set_pellets(setting.pellets.data)
                j.set_seconds(setting.seconds.data)
                j.set_confidence(setting.confidence.data)

                db['Time_Record'] = Time_Record_dict
                db.close()

                user_email = session.get("user_email")
                first_timer = setting.first_timer.data
                second_timer = setting.second_timer.data
                feeding_duration = setting.seconds.data

                schedule_feeding_alerts(first_timer, second_timer, feeding_duration, user_email)
                return redirect(url_for('dashboard'))

            elif not(6 <= first_hour <= 12):
                setting.first_timer.errors.append('First timer should be between 06:00 and 12:00 (morning to afternoon).')
                return render_template('settings.html', form=setting)
            else:
                setting.second_timer.errors.append('Second timer should be between 12:00 and 24:00 (afternoon to night).')
                return render_template('settings.html', form=setting)
        elif not re.match(pattern, setting.first_timer.data):
            setting.first_timer.errors.append('Invalid time format. Please use HH:MM format.')
            return render_template('settings.html', form=setting)
        else:
            setting.second_timer.errors.append('Invalid time format. Please use HH:MM format.')
            return render_template('settings.html', form=setting)
    else:
        Time_Record_dict = {}
        db = shelve.open('settings.db', 'r')
        Time_Record_dict = db['Time_Record']
        db.close()

        j = Time_Record_dict.get('Time_Record_Info')
        setting.first_timer.data = j.get_first_timer()
        setting.second_timer.data = j.get_second_timer()
        setting.pellets.data = j.get_pellets()
        setting.seconds.data = j.get_seconds()
        setting.confidence.data = j.get_confidence()
        return render_template('settings.html', form=setting)

def send_feeding_complete_email(user_email, feed_time):
    with app.app_context():
        try:
            msg = Message("Feeding Complete",
                          recipients=["yeapjunzhe616@outlook.com"],
                          body= f"The {feed_time} has been completed",
                          sender=("Admin", "2931801324qq@gmail.com")
                          )
            msg.body = f"The {feed_time} has been completed."
            mail.send(msg)
            print(f"Email sent to {user_email} for {feed_time}.")
        except Exception as e:
            print(f"Error sending email: {e}")

def schedule_feeding_alerts(first_timer, second_timer, feeding_duration, user_email):
    try:
        now = datetime.now()  # Use current date for scheduling
        first_timer_dt = now.replace(hour=int(first_timer[:2]), minute=int(first_timer[3:]), second=0, microsecond=0)
        second_timer_dt = now.replace(hour=int(second_timer[:2]), minute=int(second_timer[3:]), second=0, microsecond=0)

        feeding_duration = int(feeding_duration)

        # Localize the datetime objects
        timezone = pytz.timezone("Asia/Singapore")
        first_end_time = timezone.localize(first_timer_dt + timedelta(seconds=feeding_duration))
        second_end_time = timezone.localize(second_timer_dt + timedelta(seconds=feeding_duration))

        # Ensure feeding times are in the future
        if first_end_time < timezone.localize(now):
            print("First feeding time is in the past. Skipping scheduling.")
        else:
            print("Scheduling first feeding alert at:", first_end_time)
            scheduler.add_job(
                func=send_feeding_complete_email,
                trigger='date',
                run_date=first_end_time,
                args=[user_email, "first feeding complete"],
                id='first_feeding_alert',
                misfire_grace_time=3600  # Allow a 1-hour grace period for missed jobs
            )

        if second_end_time < timezone.localize(now):
            print("Second feeding time is in the past. Skipping scheduling.")
        else:
            print("Scheduling second feeding alert at:", second_end_time)
            scheduler.add_job(
                func=send_feeding_complete_email,
                trigger='date',
                run_date=second_end_time,
                args=[user_email, "second feeding complete"],
                id='second_feeding_alert',
                misfire_grace_time=3600  # Allow a 1-hour grace period for missed jobs
            )

    except ValueError as e:
        print(f"Error parsing time: {e}")
    except Exception as e:
        print(f"Scheduling error: {e}")

@app.route('/update/email', methods=['GET', 'POST'])
def update_email_settings():
    setting = emailForm(request.form)

    if request.method == 'POST' and setting.validate():
        db = shelve.open('settings.db', 'w')
        Email_dict = db['Email_Data']

        j = Email_dict.get('Email_Info')
        j.set_sender_email(setting.sender_email.data)
        j.set_recipient_email(setting.recipient_email.data)
        j.set_APPPassword(setting.App_password.data)
        j.set_days(setting.days.data)

        db['Email_Data'] =Email_dict
        db.close()
        return redirect(url_for('dashboard'))
    else:
        Email_dict = {}
        db = shelve.open('settings.db', 'r')
        Email_dict = db['Email_Data']
        db.close()

        j = Email_dict.get('Email_Info')
        setting.sender_email.data = j.get_sender_email()
        setting.recipient_email.data = j.get_recipient_email()
        setting.App_password.data = j.get_APPPassword()
        setting.days.data = j.get_days()
        return render_template('email_settings.html', form=setting)

@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error: {e}")
        return "Error generating video feed"

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    form = FeedbackForm()
    user_email = session.get('email')  # Retrieve the email from the session

    if not user_email:
        flash('Please log in to access the feedback form.', 'danger')
        return redirect(url_for('login'))

    if form.validate_on_submit():
        try:
            # Attempt to compose and send the email
            msg = Message(
                subject="New Feedback",
                sender=user_email,
                recipients=['2931801324qq@gmail.com'],
                body=f"Name: {form.name.data}\nEmail: {user_email}\nMessage:\n{form.message.data}"
            )
            mail.send(msg)

            # Flash success message and redirect to dashboard
            flash('Your feedback has been sent successfully!', 'success')
            return redirect(url_for('dashboard'))

        except Exception as e:
            # Flash error message in case of failure
            flash('An error occurred while sending your feedback. Please try again.', 'danger')
            # Log the error for debugging purposes (optional)
            app.logger.error(f'Feedback form error: {e}')

    return render_template('feedback.html', form=form)


if __name__ == '__main__':
    # Shared buffer for frames

    try:
        # Attempt to open the shelve database file for reading
        print("Attempting to open the database file for reading.")
        print("Database file opened for reading.")

        db = shelve.open('settings.db', 'r')
        # Attempt to get 'Time_Record' from db, if not found, initialize with empty dictionary
        Time_Record_dict = db.get('Time_Record',{})
        Email_dict = db.get('Email_Data', {})
        db.close()

        db = shelve.open('line_chart_data.db', 'w')
        Line_Chart_Data_dict = db.get('Line_Chart_Data',{})  # Attempt to get 'Time_Record' from db, if not found, initialize with empty dictionary
        current_date = (datetime.today()+timedelta(days=1)).strftime("%Y-%m-%d")

        if current_date not in Line_Chart_Data_dict:
            linechart = Line_Chart_Data(current_date, 0)
            Line_Chart_Data_dict[current_date] = linechart
            db['Line_Chart_Data'] = Line_Chart_Data_dict

        ###### test code ####
        today = datetime.today()
        current_date = today - timedelta(days=3)
        current_date1 = today - timedelta(days=2)
        current_date2 = today - timedelta(days=1)
        current_date3 = today

        print(current_date3,'current')

        if current_date3.strftime("%Y-%m-%d") == '2024-05-03':
            oject = Line_Chart_Data_dict.get(current_date3.strftime("%Y-%m-%d"))
            oject.set_timeRecord(0)

            oject1 = Line_Chart_Data_dict.get(current_date2.strftime("%Y-%m-%d"))
            oject1.set_timeRecord(80)

            oject2 = Line_Chart_Data_dict.get(current_date1.strftime("%Y-%m-%d"))
            oject2.set_timeRecord(300)

            oject3 = Line_Chart_Data_dict.get(current_date.strftime("%Y-%m-%d"))
            oject3.set_timeRecord(500)

        db['Line_Chart_Data'] = Line_Chart_Data_dict
        db.close()

        print('the Date you have:\n-------------------------------------------------------')
        for i in Line_Chart_Data_dict:
            print(i,': ', (Line_Chart_Data_dict.get(i).get_timeRecord()))
        print('-------------------------------------------------------')

        # Start the threads for capturing frames and processing frames
        capture_thread = threading.Thread(target=capture_frames)
        inference_thread = threading.Thread(target=process_frames)

        # Start the threads
        capture_thread.start()
        time.sleep(2.5)
        inference_thread.start()

    except:
        # If the file doesn't exist, create a new one
        print("Database file does not exist. Creating a new one.")
        db = shelve.open('settings.db', 'c')

        # create the basic setting for new user
        setting =Settings('08:30', '18:00', 1, 60,98)
        Time_Record_dict['Time_Record_Info'] = setting
        db['Time_Record'] = Time_Record_dict

        # create the basic email setup for user
        email_sender = '2931801324qq@gmail.com'
        email_password = 'felu mrhx fkkq zvdr'
        email_receiver = 'yeapjunzhe123@gmail.com'
        email_setup = Email(email_sender, email_receiver, email_password, 3)
        Email_dict['Email_Info'] = email_setup
        db['Email_Data'] = Email_dict

        # close the db
        db.close()

        #  create the line chart database
        db = shelve.open('line_chart_data.db', 'c')
        # Get today's date
        today = datetime.today()
        for i in range(7):
            # Calculate the date for the current iteration
            current_date = today - timedelta(days=i)

            # Generate data for the current date
            linechart = Line_Chart_Data(current_date, 0)

            # Store the data in the dictionary
            Line_Chart_Data_dict[current_date.strftime("%Y-%m-%d")] = linechart
        db['Line_Chart_Data'] = Line_Chart_Data_dict
        db.close()
    try:
        app.run(host='0.0.0.0', port=8080, debug=True)
    finally:
        # Stop threads on exit
        stop_event.set()
        capture_thread.join()
        inference_thread.join()

