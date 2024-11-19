import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import image assets
imagePath = 'input_image.jpg'
image2Path = 'glasses.png'
image3Path = 'pumpkin.jpg'

img = cv2.imread(imagePath) 
img2 = cv2.imread(image2Path, cv2.IMREAD_UNCHANGED)
img3 = cv2.imread(image3Path, cv2.IMREAD_UNCHANGED)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray_image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

num = 0

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

video_capture = cv2.VideoCapture(0)


def detect_bounding_box(vid, num):
    
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(50, 50))

    for (x, y, w, h) in faces:

        img = cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        if num == 0:
            image2 = image_resize(img2, width=w, height=h)
            image2_h, image2_w, _ = image2.shape

            bg = video_frame[y:y+image2_h, x:x+image2_w]

            np.multiply(bg, np.atleast_3d(255 - image2[:, :, 3])/255.0, out=bg, casting="unsafe")
            np.add(bg, image2[:, :, 0:3] * np.atleast_3d(image2[:, :, 3]), out=bg)

            video_frame[y:y+image2_h, x:x+image2_w] = bg

        else:
            image3 = image_resize(img3, width=w, height=h)
            image3_h, image3_w, _ = image3.shape

            img[y:y+image3_h, x:x+image3_w] = image3

    return faces

def mouse_click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        if num == 0:
            num = 1
        else: 
            num = 0

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame, num
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    cv2.setMouseCallback('My Face Detection Project', mouse_click, param=video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

