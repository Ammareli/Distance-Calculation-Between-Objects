from typing import Tuple
import customtkinter as ctk
import cv2 as cv
from PIL import Image, ImageTk
from detection import process, detect_object

class Gui(ctk.CTk):
    def __init__(self, fg_color: str | Tuple[str, str] | None = None, **kwargs):
        super().__init__(fg_color, **kwargs)
        self.title("Distance Detection")
        self.geometry("800x600")
        self.cam_label = ctk.CTkLabel(self,text = "")
        self.cam_label.pack(padx=20, pady = 20)
        self.button = ctk.CTkButton(self, text="My Button", command=self.button_callback)
        self.button.pack(padx=20, pady = 20)
        self.cap = cv.VideoCapture("https://192.168.1.8:8080/video")

        self.streaming()
        
    
    def button_callback(self):
        print("Button Pressed !")
    
    def check_cam(self):
        if self.cap.isOpened():
            return True
        else:
            return False
        

    def streaming(self):
        ret, img = self.cap.read()
        if self.check_cam():
            if ret:
                cv2image= cv.cvtColor(img, cv.COLOR_BGR2RGB)
                cv2DetectedImage_h = process(cv2image)
                cv2DetectedImage = detect_object(cv2DetectedImage_h)
                img = Image.fromarray(cv2DetectedImage)
                ImgTks = ImageTk.PhotoImage(image=img)
                self.cam_label.imgtk = ImgTks
                self.cam_label.configure(image=ImgTks)

            self.after(20, self.streaming)
        else:
            print("Cam Not Acessable")
    
    def on_close(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = Gui()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


