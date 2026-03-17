import os
from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, session, send_from_directory
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
import cv2
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import random

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Print debug information
print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"Static folder path: {app.static_folder}")
print(f"Static URL path: {app.static_url_path}")

# Verify images exist
images_dir = os.path.join(BASE_DIR, 'static', 'images')
print(f"Images directory: {images_dir}")
if os.path.exists(images_dir):
    print("Images directory exists")
    for img in os.listdir(images_dir):
        print(f"Found image: {img}")
else:
    print("Images directory does not exist!")
    os.makedirs(images_dir, exist_ok=True)
    print("Created images directory")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    model = models.resnet50(pretrained=False)
    num_classes = 5  # 5 yoga poses
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('E:/m0239/yoga_pose_classifier.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# Initialize model
model = load_model()

# Define the same transform as used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names with both Sanskrit and English names
CLASS_NAMES = [
    'अधोमुख श्वानासन (Adho Mukha Svanasana)',  # Downward-Facing Dog Pose
    'उत्कट कोणासन (Utkata Konasana)',          # Goddess Pose
    'फलकासन (Phalakasana)',                     # Plank Pose
    'वृक्षासन (Vrikshasana)',                   # Tree Pose
    'वीरभद्रासन-2 (Virabhadrasana II)'          # Warrior II Pose
]

# Map image files to their corresponding poses
IMAGE_TO_POSE = {
    '00000073.jpg': CLASS_NAMES[0],  # Downward-Facing Dog Pose
    '00000096.jpg': CLASS_NAMES[1],  # Goddess Pose
    '00000129.jpg': CLASS_NAMES[2],  # Plank Pose
    '00000132.jpg': CLASS_NAMES[3],  # Tree Pose
    '00000137.jpg': CLASS_NAMES[4]   # Warrior II Pose
}

# Reference images for yoga poses (all available images)
REFERENCE_IMAGES = list(IMAGE_TO_POSE.keys())

def process_frame(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    # Apply transformations
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence_value = confidence[predicted].item() * 100
        predicted_class = CLASS_NAMES[predicted.item()]
        
    return predicted_class, confidence_value

# Add these configurations after creating Flask app
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Add these new session configurations
app.config['SESSION_PROTECTION'] = 'strong'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Adjust time as needed

# Initialize SQLAlchemy and LoginManager
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Specify the login view
login_manager.login_message = 'Please log in to access this page.'  # Custom message
login_manager.login_message_category = 'info'
login_manager.session_protection = 'strong'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    has_completed_questionnaire = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Clear any existing session
    if current_user.is_authenticated:
        logout_user()
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            if not user.has_completed_questionnaire:
                return jsonify({'success': True, 'redirect': url_for('questionnaire')})
            return jsonify({'success': True, 'redirect': next_page or url_for('home')})
        return jsonify({'error': 'Invalid email or password'}), 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            
            if not username or not email or not password:
                return jsonify({'error': 'All fields are required'}), 400
            
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already exists'}), 400
                
            if User.query.filter_by(username=username).first():
                return jsonify({'error': 'Username already exists'}), 400
                
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                has_completed_questionnaire=False
            )
            db.session.add(user)
            db.session.commit()
            
            login_user(user, remember=True)  # Added remember=True for persistent login
            return jsonify({'success': True, 'redirect': url_for('questionnaire')})
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
            
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    # Clear session
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def root():
    # Clear any existing session
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)
            
        # Get the predicted class name
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence = confidence[predicted].item() * 100
        
        return jsonify({
            'class': predicted_class,
            'confidence': f'{confidence:.2f}%'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze_pose', methods=['POST'])
def analyze_pose():
    try:
        # Get the frame data from the POST request
        frame_data = request.files['frame'].read()
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get the target pose from the request
        target_pose = request.form['target_pose']
        
        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Find the index of the target pose
            target_idx = CLASS_NAMES.index(target_pose)
            
            # Get the confidence for the target pose specifically
            target_confidence = probabilities[target_idx].item() * 100
            
            return jsonify({
                'confidence': f'{target_confidence:.2f}',
                'is_matching': target_confidence > 50  # You can adjust this threshold
            })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/questionnaire')
@login_required
def questionnaire():
    return render_template('questionnaire.html')

# Add this dictionary of health conditions and their recommendations
HEALTH_RISK_RECOMMENDATIONS = {
    'heart disease': [
        'Consult your doctor before starting yoga practice',
        'Avoid inverted poses',
        'Start with gentle poses and breathing exercises',
        'Monitor your heart rate during practice',
        'Take frequent breaks'
    ],
    'high blood pressure': [
        'Avoid headstands and other inverted poses',
        'Focus on gentle standing and sitting poses',
        'Practice slow breathing exercises',
        'Monitor your blood pressure regularly',
        'Avoid holding breath during poses'
    ],
    'back pain': [
        'Focus on gentle stretching poses',
        'Avoid deep forward bends',
        'Practice cat-cow poses with caution',
        'Use props for support when needed',
        'Stop if pain increases'
    ],
    'pregnancy': [
        'Avoid hot yoga and intense practices',
        'No poses that put pressure on the abdomen',
        'Avoid lying on your back after first trimester',
        'Use props for balance and support',
        'Focus on prenatal yoga poses'
    ],
    'arthritis': [
        'Start with gentle joint movements',
        'Avoid poses that cause pain',
        'Use props for support',
        'Practice in warm conditions',
        'Focus on range of motion exercises'
    ],
    'osteoporosis': [
        'Avoid forward bends and twisting movements',
        'Focus on weight-bearing poses to build bone density',
        'Practice balance poses with support',
        'Avoid extreme spinal movements',
        'Strengthen core muscles with gentle exercises'
    ],
    'diabetes': [
        'Practice at consistent times daily',
        'Monitor blood sugar before and after practice',
        'Focus on stress-reducing poses',
        'Include gentle twists for pancreatic stimulation',
        'Maintain regular breathing patterns'
    ],
    'asthma': [
        'Focus on breathing exercises (pranayama)',
        'Practice in well-ventilated areas',
        'Keep inhaler nearby during practice',
        'Avoid inverted poses if they trigger symptoms',
        'Emphasize chest-opening poses'
    ],
    'knee problems': [
        'Use padding for kneeling poses',
        'Avoid deep knee bends',
        'Modify or skip poses that cause discomfort',
        'Focus on strengthening surrounding muscles',
        'Practice proper alignment in standing poses'
    ],
    'anxiety': [
        'Start with grounding poses',
        'Incorporate meditation and breathing exercises',
        'Practice in a quiet, calm environment',
        'Focus on restorative poses',
        'Include regular savasana practice'
    ],
    'depression': [
        'Practice energizing poses and sun salutations',
        'Include backbends and heart-opening poses',
        'Practice outdoors when possible',
        'Maintain consistent practice schedule',
        'Join group classes for social interaction'
    ],
    'fibromyalgia': [
        'Start with very gentle movements',
        'Practice in a warm environment',
        'Listen to body signals and avoid overexertion',
        'Include relaxation techniques',
        'Focus on gentle stretching and breathing'
    ],
    'multiple sclerosis': [
        'Practice in a cool environment',
        'Focus on seated and reclined poses',
        'Include balance work with proper support',
        'Take breaks when fatigue sets in',
        'Emphasize gentle stretching and breathing exercises'
    ],
    'chronic fatigue syndrome': [
        'Start with very short sessions',
        'Focus on restorative poses',
        'Practice energy conservation techniques',
        'Include meditation and breathing exercises',
        'Listen to your body and rest when needed'
    ],
    'carpal tunnel syndrome': [
        'Avoid weight-bearing on hands if painful',
        'Focus on gentle wrist stretches',
        'Use props to modify poses',
        'Practice shoulder and neck releases',
        'Avoid poses that compress the wrists'
    ],
    'vertigo': [
        'Avoid sudden head movements',
        'Practice seated poses initially',
        'Use wall support for standing poses',
        'Skip inversions and balance poses',
        'Focus on grounding practices'
    ],
    'sciatica': [
        'Avoid forward bends and twists initially',
        'Focus on gentle hip openers',
        'Practice pain-free range of motion',
        'Include modified poses that don\'t compress the spine',
        'Use props for support and alignment'
    ],
    'hip replacement': [
        'Avoid crossing legs past midline',
        'No extreme hip rotations',
        'Use props for seated poses',
        'Follow post-surgery movement restrictions',
        'Focus on gentle hip strengthening exercises'
    ],
    'epilepsy': [
        'Practice with a buddy when possible',
        'Avoid hot yoga environments',
        'Skip intense breathing exercises',
        'Modify or avoid inversions',
        'Focus on grounding poses and meditation'
    ]
}

@app.route('/submit_questionnaire', methods=['POST'])
@login_required
def submit_questionnaire():
    try:
        # Get questionnaire data
        experience = request.form.get('experience')
        fitness_level = request.form.get('fitness_level')
        goals = request.form.getlist('goals[]')
        health_conditions = request.form.get('health_conditions', '').lower()
        practice_time = request.form.get('practice_time')
        
        # Check for health risks and get recommendations
        recommendations = []
        for condition, advice in HEALTH_RISK_RECOMMENDATIONS.items():
            if condition in health_conditions:
                recommendations.extend(advice)
        
        # Update user's questionnaire status
        current_user.has_completed_questionnaire = True
        db.session.commit()
        
        response_data = {
            'success': True,
            'redirect': url_for('home')
        }
        
        # Add recommendations if any health risks were found
        if recommendations:
            response_data['recommendations'] = recommendations
            response_data['has_health_risks'] = True
        
        return jsonify(response_data)
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/get_random_poses')
def get_random_poses():
    try:
        print("Getting all poses...")  # Debug log
        images_dir = os.path.join(BASE_DIR, 'static', 'images')
        
        # Verify each image exists
        available_images = []
        for img in REFERENCE_IMAGES:
            img_path = os.path.join(images_dir, img)
            if os.path.exists(img_path):
                available_images.append(img)
                print(f"Found image: {img}")
            else:
                print(f"Missing image: {img}")
        
        if not available_images:
            print("No images found!")
            return jsonify({'error': 'No images available'}), 404
        
        # Create URLs using direct paths
        image_urls = [f'/static/images/{img}' for img in available_images]
        pose_names = [IMAGE_TO_POSE[img] for img in available_images]
        
        response_data = {
            'images': available_images,
            'image_urls': image_urls,
            'pose_names': pose_names
        }
        print(f"Returning poses data: {response_data}")  # Debug log
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in get_random_poses: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@app.route('/setup_sample_images')
def setup_sample_images():
    try:
        # Create the images directory if it doesn't exist
        images_dir = os.path.join(BASE_DIR, 'static', 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Here you would copy your sample images to the images_dir
        # For example, if you have sample images in a 'samples' folder:
        # import shutil
        # for img in ['00000073.jpg', '00000096.jpg', '00000129.jpg', '00000132.jpg', '00000137.jpg']:
        #     source = os.path.join(BASE_DIR, 'samples', img)
        #     destination = os.path.join(images_dir, img)
        #     shutil.copy2(source, destination)
        
        return jsonify({'success': True, 'message': 'Sample images set up successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Add this route for pose prediction
@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    try:
        image = None
        
        # Check if we received an image file
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Check if we received an image URL
        elif 'image_url' in request.form:
            image_url = request.form['image_url']
            # Remove the leading slash if present
            if image_url.startswith('/'):
                image_url = image_url[1:]
            # Construct the full path
            image_path = os.path.join(BASE_DIR, image_url)
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                return jsonify({'error': f'Image not found at path: {image_path}'}), 404
        
        else:
            return jsonify({'error': 'No image or image_url provided'}), 400
            
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
            
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence_value = confidence[predicted].item()
            predicted_class = CLASS_NAMES[predicted.item()]
        
        return jsonify({
            'pose_name': predicted_class,
            'confidence': confidence_value,
            'message': f'Successfully identified as {predicted_class}'
        })
        
    except Exception as e:
        print(f"Error in predict_pose: {str(e)}")  # Add debug logging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            # Create database tables
            db.create_all()
            print("Database tables created successfully!")
            print(f"Static folder path: {app.static_folder}")
            print(f"Available poses: {REFERENCE_IMAGES}")
        except Exception as e:
            print(f"Error creating database tables: {e}")
    app.run(debug=True, port=8000) 