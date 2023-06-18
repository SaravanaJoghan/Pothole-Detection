# import streamlit as st 
# from streamlit_webrtc import webrtc_streamer
# import av
# from Yolo_Preds import YOLO_Preds

# # load yolo model
# yolo = YOLO_Preds('./models/150epochs.onnx',
#                  './models/data.yaml')


# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#     # any operation 
#     #flipped = img[::-1,:,:]
#     pred_img = yolo.Predictions(img)

#     return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


# webrtc_streamer(key="example", 
#                 video_frame_callback=video_frame_callback,
#                 media_stream_constraints={"video":True,"audio":False})


    
# import streamlit as st
# import tempfile
# import os
# import shutil
# from streamlit_webrtc import webrtc_streamer
# import av
# from Yolo_Preds import YOLO_Preds
# import cv2
# import numpy as np

# # Load YOLO model
# yolo = YOLO_Preds('./models/150epochs.onnx', './models/data.yaml')

# # Function to perform object detection on a frame
# def detect_objects(frame):
#     # write necessary preprocessing steps
#     # ...
#     pred_img = yolo.Predictions(frame)
#     return pred_img

# # Streamlit app code
# def main():
#     st.title("Object Detection in Video")

#     # Upload a video file
#     video_file = st.file_uploader("Upload a video file", type=["mp4"])

#     temp_dir = tempfile.mkdtemp()  # Create temporary directory
#     output_path = os.path.join(temp_dir, "output.mp4")  # Output video file path
#     video_processed = False  # Flag to track whether video frames have been processed

#     if video_file is not None:
#         # Save the uploaded video to a temporary file
#         temp_file = os.path.join(temp_dir, "input.mp4")
#         with open(temp_file, "wb") as f:
#             f.write(video_file.read())

#         if not video_processed:
#             # Display a processing message
#             progress_text = st.empty()
#             progress_text.text("Processing video... Please wait.")

#             # Open the video file using OpenCV
#             video = cv2.VideoCapture(temp_file)

#             # Get video properties
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = video.get(cv2.CAP_PROP_FPS)
#             width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#             # Process the video frames and cache the processed frames
#             processed_frames = []
#             frame_count = 0
#             while True:
#                 # Read the next frame
#                 ret, frame = video.read()
#                 if not ret:
#                     break

#                 # Perform object detection on the frame
#                 detected_frame = detect_objects(frame)

#                 # Add the processed frame to the list
#                 processed_frames.append(detected_frame)

#                 # Update progress and display current frame number
#                 frame_count += 1
#                 progress_text.text(f"Processing frame: {frame_count}/{total_frames}")

#             # Display finish message and set video_processed flag to True
#             progress_text.text(f"Object detection completed.")
#             video_processed = True

#             # Release the video capture object
#             video.release()

#             # Create the output video file
#             output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#             # Write the processed frames to the output video
#             for frame in processed_frames:
#                 output_video.write(frame)
#             output_video.release()

#             # Clean up the uploaded video file
#             os.remove(temp_file)

#         # Provide a download link for the output video if it exists
#         if video_processed:
#             st.download_button("Download Output Video", data=open(output_path, "rb").read(), file_name="output.mp4")

# if __name__ == "__main__":
#     main()


import streamlit as st
import tempfile
import os
import shutil
from streamlit_webrtc import webrtc_streamer
import av
from Yolo_Preds import YOLO_Preds
import cv2
import numpy as np
import sys

# Load YOLO model
yolo = YOLO_Preds('./models/150epochs.onnx', './models/data.yaml')
stop_condition = False

# Function to perform object detection on a frame
def detect_objects(frame):
    # Write necessary preprocessing steps
    # ...
    pred_img = yolo.Predictions(frame)
    return pred_img

# Streamlit app code
def main():
    global stop_condition

    st.title("Object Detection in Video")

    # Upload a video file
    video_file = st.file_uploader("Upload a video file", type=["mp4"])

    temp_dir = tempfile.mkdtemp()  # Create temporary directory
    output_path = os.path.join(temp_dir, "output.mp4")  # Output video file path
    video_processed = False  # Flag to track whether video frames have been processed

    if video_file is not None:
        # Save the uploaded video to a temporary file
        temp_file = os.path.join(temp_dir, "input.mp4")
        with open(temp_file, "wb") as f:
            f.write(video_file.read())

        if not video_processed:
            # Display a processing message
            progress_text = st.empty()
            progress_text.text("Processing video... Please wait.")

            # Open the video file using OpenCV
            video = cv2.VideoCapture(temp_file)

            # Get video properties
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Process the video frames and cache the processed frames
            processed_frames = []
            frame_count = 0
            while True:
                # Read the next frame
                ret, frame = video.read()
                if not ret:
                    break

                # Perform object detection on the frame
                detected_frame = detect_objects(frame)

                # Add the processed frame to the list
                processed_frames.append(detected_frame)

                # Update progress and display current frame number
                frame_count += 1
                progress_text.text(f"Processing frame: {frame_count}/{total_frames}")

            # Display finish message and set video_processed flag to True
            progress_text.text(f"Object detection completed.")
            video_processed = True

            # Release the video capture object
            video.release()

            # Create the output video file
            output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            # Write the processed frames to the output video
            for frame in processed_frames:
                output_video.write(frame)
            output_video.release()

            # Clean up the uploaded video file
            os.remove(temp_file)

    # Provide a download link for the output video if it exists
    if video_processed:
        st.download_button("Download Output Video", data=open(output_path, "rb").read(), file_name="output.mp4")
        stop_condition = True

    if stop_condition:
        sys.exit(0)

if __name__ == "__main__":
    main()













