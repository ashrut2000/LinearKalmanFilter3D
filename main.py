import cv2
from detect import detect
from KalmanFilter import KalmanFilter
import math

def main():

    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('bounce1.mp4')
    #Create KalmanFilter object KF
    #3DKalmanFilter(dt, u_x, u_y,u_z std_acc, x_std_meas, y_std_meas,z_std_meas)

    KF = KalmanFilter(0.1, 1, 1,0, 1, 0.1, 0.1,0)
    
    #temp variable used to store old position for velocity estimation
    temp_x1=0
    temp_y1=0
    count=0
    #frame per second of the video
    fps=25
    
    


    while(True):
        # Read frame
        ret, frame = VideoCap.read()
        

        # Detect object
        centers = detect(frame)
        

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 40, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (int(x) - 45, int(y) - 45), (int(x) + 45, int(y) + 45), (255, 0, 0), 2)

            # Update
            
            (x1, y1) = KF.update(centers[0])
            v_x1=(x1-temp_x1)*fps
            v_y1=(y1-temp_y1)*fps
            temp_x1=x1
            temp_y1=y1

            # Draw a rectangle as the updated object position
            cv2.rectangle(frame, (int(x1 - 45), int(y1 - 45)), (int(x1 + 45), int(y1 + 45)), (0, 0, 255), 2)
            display=f"(velocity = ({int(v_x1)}, {int(v_y1)}) pixels/sec)"
            #displaying velocity
            cv2.putText(frame, display  ,(int(x1 + 45), int(y1 + 40)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Updated Position"  ,(10,100 ), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (10,60), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (10,20), 0, 0.5, (0,191,255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(1)


if __name__ == "__main__":
    # execute main
    main()