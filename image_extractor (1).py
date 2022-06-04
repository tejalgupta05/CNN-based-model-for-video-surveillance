from tensorflow import keras
import numpy as np
import smtplib
import cv2


model = keras.models.load_model(r"D:\ML\Mini Project CNN\secondModel(on clg images).h5")


def from_live_stream():
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()

        img = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)

        pred = np.argmax(model.predict(img))
        print(pred)

        """if pred == 1:
            cv2.imwrite('Frame.jpg', frame)
        """

        cv2.putText(frame, str(pred), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def from_video():
    cap = cv2.VideoCapture(r"D:\ML\Mini Project CNN\1.mp4")
    i = 1

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break

        cv2.imwrite(f"D:\\ML\\Mini Project CNN\\Clg Images\\1\\Frame {i}.jpg", frame)
        print(i)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def send_alert():
    sender = "rishabmamgai2001@gmail.com"
    receiver = ["harshmamgai1@gmail.com"]

    message = "check"

    try:
        obj = smtplib.SMTP('smtp.gmail.com', 587)
        obj.starttls()
        obj.login(sender, "password")

        obj.sendmail(sender, receiver, message)
        obj.quit()

    except smtplib.SMTPException as e:
        print(e)


from_live_stream()
#send_alert()
