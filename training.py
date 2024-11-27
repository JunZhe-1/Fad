import os
from roboflow import Roboflow
import subprocess

# Clone the repository and change directory
subprocess.run(["C:\\Program Files\\Git\\bin\\git.exe", "clone", "https://github.com/sovit-123/fastercnn-pytorch-training-pipeline.git"])
os.chdir("fastercnn-pytorch-training-pipeline/")

# Continue with the rest of your script...


# Install requirements
os.system("pip install -r requirements.txt")
os.system("pip install roboflow")

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="sw6kHGe5RalH1LNWh2Qi")
project = rf.workspace("fypnyplucasyuchen").project("i-fad")
version = project.version(8)
dataset = version.download("voc")


# Write data configuration YAML file
with open("data_configs/custom_data.yaml", "w") as file:
    file.write("""
TRAIN_DIR_IMAGES: 'I@FAD-8/train'
TRAIN_DIR_LABELS: 'I@FAD-8/train'
VALID_DIR_IMAGES: 'I@FAD-8/valid'
VALID_DIR_LABELS: 'I@FAD-8/valid'

CLASSES: [
  '__background__',
  'Pellets'
]

NC: 2

SAVE_VALID_PREDICTION_IMAGES: True
""")

# Get the absolute path to the home directory
home_dir = os.path.expanduser(".")

# Check if the directory exists
print(os.getcwd())
if not os.path.isdir(home_dir):
    raise FileNotFoundError(f"Directory '{home_dir}' does not exist.")

# Get the path to the home directory

# Change the current working directory to the home directory
os.chdir(home_dir)

# List the contents of the current directory (home_dir)
directory_contents = os.listdir(home_dir)
print(directory_contents)

# Install PyTorch
os.system("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))  # Print GPU name
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
# Train the model
os.system("python train.py --data data_configs/custom_data.yaml --epochs 40 --model fasterrcnn_resnet50_fpn --name custom_training --batch 16")

