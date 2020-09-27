from tkinter import ttk
from tkinter import *
from tkinter.filedialog import asksaveasfilename
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
import cv2
import datetime

dummy_file=pd.DataFrame({"FECHA":['2020-01-01', '2020-01-02'], "ACTIVIDAD":["ACCESO NO AUTORIZADO DETECTADO", "ACCESO NO AUTORIZADO DETECTADO"]})

class App:
    def __init__(self, window):

        # Config params
        
        self.model_def="config/yolov3.cfg" #path to model definition file
        self.weights_path="weights/yolov3.weights" #path to weights file
        self.class_path="data/coco.names" #path to class label file
        self.conf_thres=0.8 #object confidence threshold
        self.nms_thres=0.4 #iou thresshold for non-maximum suppression
        self.n_cpu=0 #number of cpu threads to use during batch generation
        self.img_size=416 #size of each image dimension
        self.directorio_video="" #Directorio al video
        self.checkpoint_model="" #path to checkpoint model
        self.ipcam_url = 'rtsp://admin:Cuarentena2020@192.168.100.201:554/Streaming/Channels/1'
        self.fecha_hora_ult_detectado=None
        self.alert_from_email='juanpalomino@gmail.com'
        self.alert__from_email_password='juanpalomino@gmail.com'
        self.alert_to_email='juanpalomino@gmail.com'
        

        #Window
        self.window = window
        self.window.title('DETECTOR DE USUARIOS PERMITIDOS')
        self.window.geometry('1366x768')

        # styling
        ttk.Style().configure("TButton", padding=6, relief="flat",   background="#ccc")

        # ROW 0
        self.message = Label(text='DETECTOR DE USUARIOS PERMITIDOS', fg='red')
        self.message.grid(row=0, column=0, columnspan=6, sticky=W+E)

        # ROW 1
        # Botones para elegir camara ip o archivo de video
        ttk.Button(text='DETECTAR DESDE CAMARA IP',  command=self.detect_from_ipcam).grid(
            row=1, column=0, columnspan=3, sticky=W+E)
        ttk.Button(text='DETECTAR DESDE ARCHIVO DE VIDEO',  command=self.detect_from_video).grid(
            row=1, column=3, columnspan=3, sticky=W+E)

        # ROW 2
        # Video Placeholder (comentar cuando se utilice camara ip)
        path = "./images/cctv.jpg"
        self.imageFrame = Frame(window, width=1280, height=720)
        self.imageFrame.grid(column=0, row=2, columnspan=6)
        # self.imageFrame.grid(row=0, column=0, padx=10, pady=2)

        img = ImageTk.PhotoImage(Image.open(path))
        self.panel = Label(self.imageFrame, image=img)
        self.panel.photo = img
        self.panel.grid(column=0, row=2, columnspan=6)

        # Opcion leer video de camara ip, si no funciona
        # # Graphics window

        # # Capture video frames
        # lmain = tk.Label(imageFrame)
        # lmain.grid(row=0, column=0)
        self.cap = cv2.VideoCapture(self.ipcam_url)

        

        # ROW 3
        # Botones para elegir camara ip o archivo de video
        ttk.Button(text='BITACORA DE EVENTOS',  command=self.display_log).grid(
            row=3, column=0, columnspan=2, sticky=W+E)
        ttk.Button(text='CONFIGURACIONES',  command=self.display_config).grid(
            row=3, column=2, columnspan=2, sticky=W+E)
        ttk.Button(text='ABOUT',  command=self.display_about).grid(
            row=3, column=4, columnspan=2, sticky=W+E)

    def alert(self):
        hora_actual=datetime.datetime.now()
        # Implementar bitacorizacion y alerta por email
        if self.fecha_hora_ult_detectado==None or (self.fecha_hora_ult_detectado+datetime.timedelta(seconds=60))<hora_actual:
            self.fecha_hora_ult_detectado=hora_actual
            
            with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                try:
                    server.login(self.alert_from_email, self.alert_from_email_password)
                    server.sendmail(self.alert_from_email, self.alert_to_email, "Se detecto acceso no autorizado, Fecha y Hora: {}".format(hora_actual))
                except Exception as e:
                    

    def save(self): 
        files = [('All Files', '*.*'),  
                ('CSV', '*.csv')] 
        filename = asksaveasfilename(filetypes = files, defaultextension = files) 

        # with open(filename mode='w') as myfile:
        dummy_file.to_csv(filename)
  

    def display_log(self):
        self.log_window = Toplevel()
        self.log_window.title('BITACORA')

        frame = LabelFrame(self.log_window, text="Descargar Bitacora")
        frame.grid(row=0, column=0, columnspan=3, pady=20,  padx=20)
                # Name input
        Label(frame, text="Desde: ").grid(row=1, column=0)
        self.name = Entry(frame)
        self.name.grid(row=1, column=1)
        # Price input
        Label(frame, text="Hasta: ").grid(row=2, column=0)
        self.price = Entry(frame)
        self.price.grid(row=2, column=1)

        btn = ttk.Button(frame, text = 'Save', command = self.save)
        btn.grid(row=3, column=1)
    
    def display_config(self):
        self.config_window = Toplevel()
        self.edit_window.title('CONFIGURACION')

    def display_about(self):
        self.about_window = Toplevel()
        self.edit_window.title('ABOUT')

    def detect_from_ipcam(self):
        _, frame = self.cap.read()
        # if _ ==True:
        try:
            frame=cv2.resize(frame,(1300,720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            # else:
            #     break
            # frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk)
            # self.panel.pack()
        except Exception as e:
            print(e)
        self.panel.after(10, self.detect_from_ipcam)

    def detect_from_video():
        pass


if __name__ == "__main__":
    window = Tk()
    app = App(window)
    window.mainloop()
