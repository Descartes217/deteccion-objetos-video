from tkinter import ttk
from tkinter import *
from tkinter.filedialog import asksaveasfilename
from PIL import ImageTk, Image
import numpy as np
import pandas as pd
import cv2
import datetime, time
import sqlite3
from threading import Thread

dummy_file=pd.DataFrame({"FECHA":['2020-01-01', '2020-01-02'], "ACTIVIDAD":["ACCESO NO AUTORIZADO DETECTADO", "ACCESO NO AUTORIZADO DETECTADO"]})

class ThreadedVideo():
    def __init__(self, src, detection_params):
        # Detection 
        self.detection_params=detection_params
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(self.detection_params["model_def"], img_size=self.detection_params["img_size"]).to(self.device)

        
        if self.detection_params["weights_path"].endswith(".weights"):
            model.load_darknet_weights(self.detection_params["weights_path"])
        else:
            model.load_state_dict(torch.load(self.detection_params["weights_path"]))

        model.eval()  
        classes = load_classes(self.detection_params["class_path"])
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.a=[]
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)
    
    def return_frame(self):
        self.frame=cv2.resize(self.frame,(848,480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        self.frame, alert =self.detect(self, self.frame)
        return self.frame, alert

    def detect(self,frame):

        flag_no_autorizado=False

        RGBimg=Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        with torch.no_grad():
            detections = self.model(imgTensor)
            detections = non_max_suppression(detections, self.detection_params["conf_thres"], self.detection_params["nms_thres"])


        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, self.detection_params["img_size"], RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)
                    cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)# Nombre de la clase detectada
                    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 3) # Certeza de prediccion de la clase

                    if classes[int(cls_pred)]='no_autorizado':
                        flag_no_autorizado=True
        
        # esta pendiente reconvertira RBG
        return frame, flag_no_autorizado

    def Convertir_RGB(img):
        # Convertir Blue, green, red a Red, green, blue
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        return img


    def Convertir_BGR(img):
        # Convertir red, blue, green a Blue, green, red
        r = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        b = img[:, :, 2].copy()
        img[:, :, 0] = b
        img[:, :, 1] = g
        img[:, :, 2] = r
        return img

class App:
    def __init__(self, window):

        # Config params
        self.detection_params={

        "model_def"="config/yolov3.cfg" #path to model definition file
        ,"weights_path"="weights/yolov3.weights" #path to weights file
        ,"class_path"="data/coco.names" #path to class label file
        ,"conf_thres"=0.8 #object confidence threshold
        ,"nms_thres"=0.4 #iou thresshold for non-maximum suppression
        ,"n_cpu"=0 #number of cpu threads to use during batch generation
        ,"img_size"=416 #size of each image dimension
        ,"directorio_video"="" #Directorio al video
        ,"checkpoint_model"="" #path to checkpoint model
        }

        self.ipcam_url = 'rtsp://admin:Cuarentena2020@192.168.100.201:554/Streaming/Channels/1'
        self.video_url = 'videos/1.mp4'
        self.fecha_hora_ult_detectado=None
        self.alert_from_email='alertas.rtsb@gmail.com'
        self.alert__from_email_password='Cuarentena.2020'
        self.alert_to_email='alertas.rtsb@gmail.com'
        self.db='example.db'

        self.video_thread=None

        conn = sqlite3.connect('example.db')

        #Window
        self.window = window
        self.window.title('DETECTOR DE USUARIOS PERMITIDOS')
        self.window.geometry('855x590')

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
        path = "./images/placeholder.png"
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
                    server.sendmail(self.alert_from_email, self.alert_to_email, "Se detecto acceso no autorizado, Fecha y Hora: {}".format(hora_actual.strftime("%d/%m/%Y %H:%M:%S")))
                    server.close()
                except Exception as e:
                    print("Error al enviar el mail")
            
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                c=conn.cursor
                c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='bitacora_accesos' ''')

                if c.fetchone()[0]==1:
                    c.execute('''INSERT INTO bitacora_accesos VALUES ('{}', '{}') '''.format(hora_actual.strftime("%d/%m/%Y %H:%M:%S"), "ACCESO NO AUTORIZADO"))
                # print(sqlite3.version)
                else:
                    c.execute(''' CREATE TABLE bitacora_accesos  (FECHA_HORA TEXT,EVENTO TEXT) ''')
                    c.execute('''INSERT INTO bitacora_accesos VALUES ('{}', '{}') '''.format(hora_actual.strftime("%d/%m/%Y %H:%M:%S"), "ACCESO NO AUTORIZADO"))
                
                conn.commit()
            
            except Error as e:
                print(e)
            
            finally:
                if conn:
                    conn.close()

            

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
        if self.video_thread==None:
            self.video_thread=ThreadedVideo(ipcam_url)
        # _, frame = self.cap.read()
        # if _ ==True:
        try:
            # frame=cv2.resize(frame,(1300,720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            frame=self.video_thread.return_frame()
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

    def detect_from_video(self):
        if self.video_thread==None:
            self.video_thread=ThreadedVideo(self.video_url)
        # _, frame = self.cap.read()
        # if _ ==True:
        try:
            # frame=cv2.resize(frame,(1300,720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            frame=self.video_thread.return_frame()
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


if __name__ == "__main__":
    window = Tk()
    app = App(window)
    window.mainloop()
