import torch
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import threading
from datetime import datetime, timedelta

if torch.cuda.is_available():
    print('you are using gpu to process the video camera')
else:
    print('no gpu is found in this python environment. using cpu to process')

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        self.cond = threading.Condition()
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
            with self.cond:
                self.frame = img if rv else None
                self.pellets_num = counter
                self.cond.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        with self.cond:
            if wait:
                # If seqnumber is not provided, get the next sequence number
                if seqnumber is None:
                    seqnumber = self.pellets_num + 1
                if seqnumber < 1:
                    seqnumber = 1

                # Wait until the latest frame's sequence number is greater than or equal to seqnumber
                rv = self.cond.wait_for(lambda: self.pellets_num >= seqnumber, timeout=timeout) # if there is a pellets. should get "true"
                if not rv:
                    return (self.pellets_num, self.frame)  # Return the latest frame if timeout occurs
            return (self.pellets_num, self.frame)  # Return the latest frame


# define the id "1" for pellets
# do note that in the pth file, the pellet id also is 1
class_labels = {
        1: 'Pellets'
    }

# pth file where you have defined on roboflow
model_path = './best_model.pth'


# Define the create_model function here
def create_model(num_classes, pretrained=False, coco_model=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
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


def main():
    # open some camera
    cap = cv2.VideoCapture('rtsp://admin:Citi123!@192.168.1.64:554/Streaming/Channels/101')
    #cap =cv2.VideoCapture('./sample.mp4')

    #cap =cv2.VideoCapture(0)


    cap.set(cv2.CAP_PROP_FPS, 30)

    # wrap i
    fresh = FreshestFrame(cap)

    # Load the Faster R-CNN model from the .pth file
    num_classes = 2  # Assuming 2 classes for 'Pellets' and background
    model = load_model(model_path, num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # define the dictionary to store the number of pellets
    # Assuming 1 class for 'Pellet'
    object_count = {1: 0}

    feeding = False
    feeding_timer = None

    first_feeding_time = 15
    feeding_minutes_min = 29


    second_feeding_time = 15
    second_feeding_time_min = 31


    showing_timer = None
    desired_time = None

    formatted_desired_time = None
    current_datetime = datetime.now()

    while True:

        # Process the predictions and update object count
        temp_object_count = {1: 0}  # Initialize count for the current frame


        current_time = datetime.now().time()
        if (current_time.hour == first_feeding_time or current_time.hour == second_feeding_time) and (current_time.minute == feeding_minutes_min or current_time.minute == second_feeding_time_min) and current_time.second == 0:
            feeding = True
            feeding_timer = None
            showing_timer = None
            # round = 2 if current_time.hour == first_feeding_time and current_time.minute == feeding_minutes_min else 1
            # if round == 2:
            #     desired_time = current_datetime.replace(hour=second_feeding_time, minute=feeding_minutes_min, second=0,
            #                                             microsecond=0)
            #     formatted_desired_time = desired_time.strftime("%I:%M %p")
            #
            # else:
            #
            #     # Add one day to the current date and time
            #     next_day = current_datetime + timedelta(days=1)
            #     # Set desired_time to 8 AM of the next day
            #     desired_time = next_day.replace(hour=first_feeding_time, minute=feeding_minutes_min, second=0, microsecond=0)

        cnt, frame = fresh.read(seqnumber=object_count[1] + 1)
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

                # 0.95 is the highest, while we are looking for 90% of the probability
                if score > 0.975:
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) #(0,255,0) is the color (blue, green, yellow)
                    cv2.putText(frame, f'{class_labels[label]}: {score:.2f}', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    temp_object_count[label] += 1


                    # Start feeding timer if pellets are detected
                    if label == 1 and feeding_timer is None and feeding:
                        feeding_timer = time.time()

        # store the pellets number to the object count which is permanently
        for label, count in temp_object_count.items():
            object_count[label] = count

        # Check feeding timer and switch to stop feeding if required
        if feeding_timer is not None and feeding:
            elapsed_time = (time.time() - feeding_timer)

            print( f'elapsed time: {elapsed_time:.3f}' )

            if elapsed_time > 60 and sum(object_count.values()) > 3:
                feeding = False
                feeding_timer = None
                showing_timer = time.time()

            elif object_count[1] == 0: # error prevention
                feeding_timer = None


        # Display the frame with detections and object count
        for label, count in object_count.items():
            text = f'{class_labels[label]} Count: {count}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_position = (frame.shape[1] - text_size[0] - 10, 30 * label)
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display feeding or stop feeding text just below the object counter
        text_position_feed = (frame.shape[1] - text_size[0] - 10  , 30 * (max(object_count.keys()) + 1))
        round_position = (frame.shape[1] - 200 - 50, 30 * (1 + 1))

        if feeding:
            cv2.putText(frame, "Feeding...", text_position_feed,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if showing_timer is not None:
                i = time.time() - showing_timer
                if i > 3:
                    showing_timer = None
                    print('running')
                else:
                    cv2.putText(frame, "Stop Feeding", text_position_feed,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:

                if current_time.hour <= first_feeding_time and current_time.minute <= feeding_minutes_min:
                    desired_time = current_datetime.replace(hour=first_feeding_time, minute=feeding_minutes_min, second=0,
                                                            microsecond=0)
                    formatted_desired_time = desired_time.strftime("%I:%M %p")
                    cv2.putText(frame, "next round: " + formatted_desired_time, round_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


                elif (current_time.hour <= second_feeding_time and current_time.minute <= second_feeding_time_min):
                    desired_time = current_datetime.replace(hour=second_feeding_time, minute=second_feeding_time_min, second=0,
                                                            microsecond=0)
                    formatted_desired_time = desired_time.strftime("%I:%M %p")
                    cv2.putText(frame, "next round: " + formatted_desired_time, round_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    # Add one day to the current date and time
                    next_day = current_datetime + timedelta(days=1)
                    # Set desired_time to 8 AM of the next day
                    desired_time = next_day.replace(hour=first_feeding_time, minute=feeding_minutes_min, second=0, microsecond=0)

                    formatted_desired_time = desired_time.strftime("%I:%M %p")

                    cv2.putText(frame, "next round: " + formatted_desired_time, round_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.namedWindow('Pellets Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Pellets Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fresh.stop()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
