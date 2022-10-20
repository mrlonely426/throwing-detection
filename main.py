

from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QFileDialog
from PyQt5.QtGui import QPixmap,QPainter,QIcon,QImage
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from throw import Ui_MainWindow

import cv2
import numpy as np
import json
import os
import time
import torch
import sys

from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam,LoadStreams
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_imshow,check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr,crop_box
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt,colors, plot_one_box, plot_one_box_PIL,plot_line_box
from utils.torch_utils import select_device, time_synchronized
#from utils.capnums import Camera
from dialog.rtsp_win import Window
from utils.dialog_message import MessageBox




COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

class DetThread(QThread):
    send_raw = pyqtSignal(np.ndarray)
    send_out = pyqtSignal(np.ndarray)

    send_percent = pyqtSignal(int)
    send_FPS = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.source='./back_img/test.mp4'
        self.weights = './runs/train_v5/exp_5m_1280_train(day)_valid(all)/weights/best.pt'
        self.current_weight=''
        self.percent_length = 1000
        self.k_size=3
        self.min_area=20
        self.rate_check = True
        self.rate = 100
        self.jump_out = False
        self.is_continue = True
        self.conf_thres = 0.25
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.save_time_length=20
        self.save_img = True
        self.save_dir = Path(os.path.join(os.path.dirname(__file__),'runs/detect'))

    @torch.no_grad()
    def run(self,
            imgsz=1280,
            max_det=1000,
            device='',
            view_img=True,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            nosave=False,
            classes=None,
            agnostic_nms=False,
            augment=False,
            visualize=False,
            update=False,
            project='runs/detect',
            name='exp',
            exist_ok=False,
            line_thickness=3,
            hide_labels=False,
            hide_conf=False,
            half=False,
            ):
        try:
            #初始化
            device = select_device(device)
            half &= device.type != 'cpu'

            model = attempt_load(self.weights, map_location=device)

            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            #print(stride)
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                #cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            # 跳帧检测
            jump_count = 0
            start_time = time.time()
            plot_boxs = []
            conf_time = True
            time_save = 0
            count_save = 0

            vid_writer = None
            dataset = iter(dataset)
            crop = None
            sig_save = False
            sig_tra_img = False
            video_path = None
            save_once = True
            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    sig_tra_img = True
                    print("停止")
                    #self.send_msg.emit('停止')
                    break
                # 临时更换模型
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                # 暂停开关
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    print(path)
                    img_width =im0s.shape[1]
                    img_height = im0s.shape[0]
                    print('img_width:{},img_height:{}'.format(img_width,img_height))
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1

                    # 每三十帧刷新一次输出帧率
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_FPS.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Process detections
                    #label = None

                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):

                                plot_boxs.append(torch.tensor(xyxy).view(1,4).numpy().tolist()[0])
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加
                                #plot_one_box(xyxy, im0, label=label, color=colors(c, True),line_thickness=line_thickness)
                                #save_im0=im0s.copy()
                                print(len(plot_boxs))
                                print('count:{},conf:{}'.format(count,conf))
                                plot_line_box(plot_boxs,im0, label=label, color=colors(c, True),line_thickness=line_thickness,on_box=True)
                                crop = crop_box(xyxy, im0s,scale=img_width/img_height)
                                if float(conf) > 0.2 and conf_time:
                                    time_save = time.time()
                                    count_save = 1
                                    conf_time =False

                                #print('第一次出现抛物时间：{}，-----距离第一次出现抛物时长是:{}'.format(time_save,time.time()-time_save))

                        else:
                            print('未检测到物体')
                            crop = crop

                    print('第一次出现抛物时间：{}，-----距离第一次出现抛物时长是:{}'.format(time_save, time.time() - time_save))
                    if time_save != 0:
                        count_save += 1
                        if count_save < self.save_time_length :
                            sig_save = True

                        else:
                            sig_save = False


                        if count_save > self.save_time_length * 1.5 :
                            sig_tra_img = True

                            print('保存图片')
                        else:
                            sig_tra_img = False

                    if sig_tra_img and self.save_img and save_once:
                        img_tra=im0s.copy()
                        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                        plot_line_box(plot_boxs, img_tra, label=None, color=colors(200, True),
                                      line_thickness=line_thickness, on_box=False)
                        #cv2.imwrite('F://Project//pycharm//weather//runs//detect//tra_img.jpg', img_tra)
                        cv2.imwrite(str(self.save_dir.joinpath(now + '-tra_img.jpg')), img_tra)
                        save_once = False

                    if sig_save and self.save_img:
                        print('sig_save:{}'.format(sig_save))
                        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                        if dataset.mode =='image':
                            cv2.imwrite(str(self.save_dir.joinpath(now+'-tra_img.jpg')),im0)
                        else:
                            save_path = self.save_dir.joinpath(now+'-tra_video.mp4')
                            print(str(save_path))
                            if video_path !=save_path and video_path is None:
                                video_path =save_path
                                if isinstance(vid_writer,cv2.VideoWriter):
                                    vid_writer.release()

                                if self.vid_cap:
                                    fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:
                                    fps, w, h = 30,im0.shape[1],im0.shape[0]

                                vid_writer = cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'), int(fps / 6), (w, h))
                                #vid_writer = cv2.VideoWriter('F://Project//pycharm//weather//runs//detect//save_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),int(fps/4),(w,h))

                            vid_writer.write(im0)
                            print('保存视频')




                    # 控制视频发送频率
                    if self.rate_check:
                        time.sleep(1/self.rate)
                    #print(type(im0s))
                    self.send_out.emit(im0)
                    if crop is not None:
                        self.send_raw.emit(crop)
                    else:
                        self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    #self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])

                    #print(statistic_dic)
                    #self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        #self.send_msg.emit('检测结束')
                        print('检测结束')
                        # 正常跳出循环
                        break

        except Exception as e:
            print('message:{}'.format(e))
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)
            #self.send_msg.emit('%s' % e)




class myMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(myMainWindow,self).__init__(parent)
        self.setupUi(self)

        self.comboBox.clear()
        self.pt_list = os.listdir('./runs/train_v5')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./runs/train_v5/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        self.comboBox.currentTextChanged.connect(self.change_model)

        self.play_pause_Button.clicked.connect(self.play_pause)
        self.play_pause_Button.setCheckable(True)

        self.videoButton.clicked.connect(self.open_file)
        self.rtspButton.clicked.connect(self.chose_rtsp)
        self.min_Button.clicked.connect(self.showMinimized)
        self.max_Button.clicked.connect(self.max_windows)
        self.close_Button.clicked.connect(self.close)

        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./runs/train_v5/%s" % self.model_type #"./runs/train_v5/exp_5m_1280_train(day)_valid(all)/weights/best.pt"
        self.det_thread.source = './back_img/test.mp4'       #'./back_img/test.mp4'
        #self.det_thread.current_source='./back_img/test.mp4'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_label))
        self.det_thread.send_out.connect(lambda x: self.show_image(x, self.dete_out_label))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_FPS.connect(lambda  x:self.FPS_label.setText(x))

        self.time_spinBox.valueChanged.connect(lambda x:self.change_val(x,'time_spinBox'))
        self.time_horizontalSlider.valueChanged.connect(lambda x:self.change_val(x,'time_horizontalSlider'))

        self.stop_Button.clicked.connect(self.stop)

    def change_val(self, x, flag):
        if flag == 'time_spinBox':
            self.time_horizontalSlider.setValue(int(x*4))
        elif flag == 'time_horizontalSlider':
            self.time_spinBox.setValue(x*4)
            self.det_thread.save_time_length = x*4
        else:
            pass


    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.okkButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_Button, title='提示', text='请稍等，正在加载rtsp视频流', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            #self.statistic_msg('加载rtsp：{}'.format(ip))
            print('加载rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            print('%s' % e)
            #self.statistic_msg('%s' % e)

    def play_pause(self):
        self.det_thread.jump_out = False
        #print(self.play_pause_Button.isChecked())
        if self.play_pause_Button.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
                print('播放')
            source = os.path.basename(self.det_thread.source)
            source = '摄像头设备' if source.isnumeric() else source
            print('正在检测 >> 模型：{}，文件：{}'.format(os.path.basename(self.det_thread.weights),source))
            #self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.format(os.path.basename(self.det_thread.weights),source))
        else:
            self.det_thread.is_continue = False
            print("暂停")


    def open_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")

        if name:
            self.det_thread.source = name
            #self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            print("切换视频为:{}".format(self.det_thread.source))
            self.stop()

    def stop(self):
        self.det_thread.jump_out = True


    def max_windows(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def search_pt(self):
        pt_list = os.listdir('./runs/train_v5')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./runs/train_v5/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./runs/train_v5/%s" % self.model_type
        #self.statistic_msg('模型切换为%s' % x)
        print('模型切换为%s' % x)


    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))



if __name__ == '__main__':
 app = QApplication(sys.argv)
 mywindow = myMainWindow()
 mywindow.show()
 sys.exit(app.exec_())

