import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template,redirect ,session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import bcrypt

app = Flask(__name__)  

# Configure the SQLAlchemy database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Initialize SQLAlchemy
db = SQLAlchemy(app)
# Create a model for the User table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

with app.app_context():
    db.create_all()

# Load the binary classification model (tumor yes or no)
binary_model = load_model('BrainTumor10EpochsCategorical.h5')

# Load the multi-class tumor classification model
multi_model = load_model('MultiClassTumorModel.h5')

print('Models loaded. Check http://127.0.0.1:5000/')

def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"

def get_tumor_type_name(tumor_type):
    if tumor_type == 0:
        return "Glioma"
    elif tumor_type == 1:
        return "Meningioma"
    elif tumor_type == 2:
        return "Pituitary"

def get_result(img_path):
    image = cv2.imread(img_path)
    print("File Path:", img_path)
    if image is not None:
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0)

        # Use the binary model to check if a tumor is detected
        binary_result = binary_model.predict(input_img)

        if binary_result[0][1] > 0.5:
            # If a tumor is detected, determine the tumor type using the multi-class model
            multi_result = multi_model.predict(input_img)
            tumor_type = np.argmax(multi_result)
            # return f"Tumor Detected: {get_tumor_type_name(tumor_type)}"
            return f"Tumor Detected: Yes"
        else:
            return "No Tumor Detected"
    else:
        return "Error: Image not found or couldn't be loaded"


def get_tumor_type(img_path):
    image = cv2.imread(img_path)
    print("File Path:", img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    binary_result = binary_model.predict(input_img)

    if binary_result[0][1] > 0.5:
        multi_result = multi_model.predict(input_img)
        tumor_type = np.argmax(multi_result)
        return get_tumor_type_name(tumor_type)
    else:
        return "No Tumor Detected"
@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()
    
@app.route('/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return "Passwords do not match"

        # Hash the password before storing it
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create a new User instance with the hashed password and add it to the database
        new_user = User(username=username, email=email, password=hashed_password.decode('utf-8'))
        db.session.add(new_user)
        db.session.commit()

        # Redirect to the homepage after successful signup
        return redirect('/')

    return "Invalid request"


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Query the database to check if the user exists with the provided username
        user = User.query.filter_by(username=username).first()

        if user:
            # Verify if the provided password matches the hashed password in the database
            hashed_password = user.password  # Fetch the hashed password from the database
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                # Set the user session to indicate the user is logged in
                session['logged_in'] = True
                session['username'] = username
                return render_template('index.html', success_message="Logged in successfully!")

        # If username or password is invalid, show an alert on the login page
        return render_template('login.html', error_message="Invalid username or password. Please try again.")

    # If it's a GET request or unsuccessful login attempt, render the login page
    return render_template('login.html')


# Example route that requires authentication
@app.route('/protected_route')
def protected_route():
    if not session.get('logged_in'):
        # Redirect to login page or handle unauthorized access
        return redirect('/')
    
    # Continue with the functionality for the protected route
    return render_template('protected_page.html')
#TODO-----------------



@app.route('/logout')
def logout():
    db.session.remove()
    return redirect('/')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle any POST request data if needed
        pass
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = get_result(file_path)
        return result
    return "No file uploaded"


@app.route('/get_tumor_type', methods=['POST'])
def get_tumor_type_endpoint():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        result = get_tumor_type(file_path)
        return result
    return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
