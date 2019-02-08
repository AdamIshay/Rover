import cv2                            # importing Python OpenCV
import imutils


class rover_tracker:
    def __init__(self):
        self.delta_thresh = 4
        self.min_area = 150                     # Threshold for triggering "motion detection"
        self.cam = cv2.VideoCapture(0)             # Lets initialize capture on webcam
        self.avg = None

    def get_frame(self):
        self.frame = self.cam.read()[1]

    def track(self):
        grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (21, 21), 0)

        # if the average frame is None, initialize it
        if self.avg is None:
            self.avg = grey.copy().astype("float")
            pass

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(grey, self.avg, 0.5)
        frameDelta = cv2.absdiff(grey, cv2.convertScaleAbs(self.avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, self.delta_thresh, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        if len(cnts) == 0:  # if no contours return None
            return None
        else:
            for idx, c in enumerate(cnts):
                # if the contour is too small, ignore it
                # print cv2.contourArea(c)
                if cv2.contourArea(c) < self.min_area:
                    return None

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)

                # these if statements ensure that the tracker doesn't pick up
                # on any differences outside of the track.
                if x < 170:
                    return None

                if y < 125 and x > 402:
                    return None
                elif y < 235 and x < 410:
                    return None

                # get the center of the bounding rectangle
                cx, cy = (x + w) // 2, (y + h) // 2

            return [cy, cx]



tracker = rover_tracker()

while True:
    tracker.get_frame()
    
    pdb.set_trace()
    center = tracker.track()
    print(center)
