from tkinter import *
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import cv2
import os
import sys
import numpy as np
import time

ROOT_DIR = os.path.abspath("")
MASK_RCNN_DIR = os.path.join(ROOT_DIR,'Mask_RCNN')
SORT_DIR = os.path.join(ROOT_DIR,'Sort')
MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_traffic_0107.h5')

sys.path.append(ROOT_DIR)
sys.path.append(MASK_RCNN_DIR)
sys.path.append(SORT_DIR)

from traffic.utils.TrafficConfig import TrafficConfig
from traffic.utils.Predictor import Predictor

class Root(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title = "Vehicle Detector"
        self.geometry("600x250")

        self.buttonIn = ttk.Button(text="Открыть", command=self.selectInputFile)
        self.buttonOut = ttk.Button(text="Сохранить", command=self.selectOutputFile)
        self.buttonStart = ttk.Button(text="Начать", command=self.startPredictor)

        self.labelInputName  = ttk.Label(text="", width=80, wraplength = 540)
        self.labelOutputName = ttk.Label(text="", width=80, wraplength = 540)
        self.labelInfo = ttk.Label(text="Выберите входнной и выходной файл",width=80, wraplength = 540)
        self.progressbar = ttk.Progressbar(length = 300)

        self.buttonIn.grid(row=1,column = 0)
        self.buttonOut.grid(row=2,column=0)
        self.buttonStart.grid(row=3,column=0)

        self.labelInputName.grid(row=1,column=1)
        self.labelOutputName.grid(row=2,column=1)
        self.labelInfo.grid(row=4, column=0, columnspan=2)

        self.progressbar.grid(row=5,column=0,columnspan=2)

        self.input_file_name = None
        self.output_file_name = None

        config = TrafficConfig()
        self.predictor = Predictor(mode="inference",
                           model_dir=ROOT_DIR,
                           model_path=MODEL_PATH,
                           config=config,
                           max_age=30,
                           min_hits=15)

    def selectInputFile(self):
        self.input_file_name = fd.askopenfilename()
        self.labelInputName['text']=self.input_file_name

    def selectOutputFile(self):
        self.output_file_name = fd.asksaveasfilename(filetypes=(("AVI files", "*.avi"),))
        if self.output_file_name[len(self.output_file_name)-4:] != '.avi':
            self.output_file_name += '.avi'
        self.labelOutputName['text']=self.output_file_name

    def startPredictor(self):
        if self.input_file_name is None or self.output_file_name is None:
            mb.showerror("Ошибка", "Введите имя входного и выходного файла")
        elif mb.askokcancel(title="Message", message="Распознование использует библиотеку TensorFlow, которая будет использовать ресурсы компьютера."
                                                     "Подготовка и выполнение задачи при помощи данной программы может занять продолжительное время." 
                                                    "Вы точно хотите продолжить?"):
            self.labelInfo['text']='Считывание данных и подготовка к работе'
            self.labelInfo.update()

            stream = cv2.VideoCapture(self.input_file_name)
            frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

            self.progressbar['maximum']=frame_count
            self.progressbar['value']=0

            w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'w = {w}, h = {h}')
            mask_path = 'mask.png'
            main_mask = cv2.imread(mask_path)
            main_mask = cv2.resize(main_mask, dsize=(w, h))
            main_mask = main_mask.astype(np.bool)

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            writer = cv2.VideoWriter(self.output_file_name,fourcc, 25, (w, h), True)

            self.labelInfo['text']='Запуск распознования'
            self.labelInfo.update()
            self.predictor.load_instream(stream)
            self.predictor.load_outstrean(writer)
            self.predictor.load_mask(main_mask)
            i=1
            total_time = 0
            avg_time = 0
            while self.predictor.streamIsOpened():
                start_time = time.time()
                self.predictor.do_predict()
                predict_time = time.time()-start_time
                total_time += predict_time
                if i>1:
                    avg_time = (avg_time*(i-1)+predict_time)/i
                remaining_time = avg_time*(frame_count-i)
                self.labelInfo['text']=f'Кадр {i} из {frame_count}. Времени прошло: {round(total_time,2)} c. Времени осталось: {round(remaining_time,2)} c.'
                self.progressbar['value']=i
                self.progressbar.update()
                i+=1
            self.labelInfo['text']='Сохранение результатов'
            self.labelInfo.update()
            writer.release()
            stream.release()
            if mb.askokcancel(title = "Завершено", message=f'Распознование завершено. Результат доступен по адрессу {self.output_file_name}'):
                try:
                    os.system(f'explorer "{os.path.dirname(self.output_file_name)}/"')
                except:
                    print('finish')
            self.labelInfo['text'] = 'Выберите входнной и выходной файл'
            self.progressbar['value']=0
            self.labelInfo.update()
if __name__=="__main__":
    root = Root()
    root.mainloop()