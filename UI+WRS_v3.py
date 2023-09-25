import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5 import uic
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import fdasrsf as fs
import numpy as np

form_main = uic.loadUiType('UI+WRS_v3.ui')[0]


class Mainwindow(QMainWindow, form_main):

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('KIMS_emblem01.png'))
        self.setupUi(self)
        self.textBrowser.append("                    --UQ Program for WRS profile--")
        self.textBrowser.append(" ")
        self.textBrowser.append("                  Made by KIMS (Jeong Seong Gyun)")
        self.initUI_Main()

    def initUI_Main(self):
        # btn connect
        self.btn_load.clicked.connect(self.loadfile)
        self.btn_savefig.clicked.connect(self.savefig)
        self.btn_savewarp.clicked.connect(self.savewarp)
        self.btn_savesample.clicked.connect(self.savesample)
        self.btn_saveUQ.clicked.connect(self.saveUQ)

        self.btn_warp.clicked.connect(self.warping)
        self.btn_sample.clicked.connect(self.sampling)
        self.btn_UQ.clicked.connect(self.UQ)
        self.btn_cal.clicked.connect(self.calculate)

        self.btn_warpshow.toggled.connect(self.warpplot_toggled)

        self.chkbox_Kmean.stateChanged.connect(self.Kplot)
        self.chkbox_cb.stateChanged.connect(self.replot)
        self.chkbox_mean.stateChanged.connect(self.replot)
        self.chkbox_tb.stateChanged.connect(self.replot)
        self.chkbox_WRSmean.stateChanged.connect(self.replot)

        self.spinBox_4.valueChanged.connect(self.warpplot_toggled)

        # introduction(Made by KIMS) erase
        self.init = 0

        # Figure canvas initialization
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.graph_verticalLayout.addWidget(self.canvas)

        self.show()

    def loadfile(self):
        # get filename from dialog
        fname_load = QFileDialog.getOpenFileName(self, '', '', 'Excel(*.xlsx *xls)')

        # erase introduction when load first time
        if self.init == 0:
            self.init = 1
            self.textBrowser.clear()

        # clear canvas & add
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        try:
            # read database
            db = pd.read_excel(fname_load[0])
            self.db_np = db.to_numpy()

            # get self.x, self.y per function
            for i in range(0, int(self.db_np.shape[1]/2)):
                i = i+1
                # x1 = db["1x"]
                globals()['x' + str(i)] = db[str(i) + "x"]
                # y1 = db["1y"]
                globals()['y' + str(i)] = db[str(i) + "y"]

                # drop invalid data
                # x1 = x1.dropna()
                exec("self.x%s = x%s.dropna()" % (str(i), str(i)))
                # y1 = y1.dropna()
                exec("self.y%s = y%s.dropna()" % (str(i), str(i)))

                # plot individually
                # ax.plot(self.x1, self.y1)
                exec("ax.plot(self.x%s, self.y%s)" % (str(i), str(i)))

                # plotting x, y on canvas
                ax.grid()
                ax.set_xlabel("Normalized Depth from ID (r/t)")
                ax.set_ylabel("Residual Stress")
                ax.set_title("raw data graph")
                self.canvas.draw()

                # -load- v
                self.btn_warp.setEnabled(True)
                self.spinBox.setEnabled(True)
                self.exportlabel.setEnabled(True)
                self.btn_savefig.setEnabled(True)
                # ----------------------------------------
                # -warping- x
                self.graphlabel.setDisabled(True)
                self.btn_sample.setDisabled(True)
                self.spinBox_2.setDisabled(True)
                self.spinBox_3.setDisabled(True)
                self.btn_warpshow.setDisabled(True)
                self.chkbox_Kmean.setChecked(False)
                self.chkbox_Kmean.setDisabled(True)
                self.btn_savewarp.setDisabled(True)
                # ----------------------------------------
                # -Sampling- x
                self.btn_UQ.setDisabled(True)
                self.btn_savesample.setDisabled(True)
                # ----------------------------------------
                # -UQ- x
                self.btn_cal.setDisabled(True)
                self.chkbox_mean.setChecked(False)
                self.chkbox_mean.setDisabled(True)
                self.chkbox_cb.setChecked(False)
                self.chkbox_cb.setDisabled(True)
                self.chkbox_tb.setChecked(False)
                self.chkbox_tb.setDisabled(True)
                self.chkbox_WRSmean.setDisabled(True)
                self.btn_saveUQ.setDisabled(True)

                # bundle select
                self.btn_bundleselect.setDisabled(True)
                self.spinBox_4.setDisabled(True)
        except:
            self.textBrowser.append("improper input.\n please check the file.")

    def savefig(self):
        # get filename from dialog
        savefname = QFileDialog.getSaveFileName(None, "Select destination folder and file name", ""
                                                     , "Image files (*.png *jpg)")[0]
        if savefname != '':
            self.fig.savefig(savefname)
            QMessageBox.information(self, 'Message Box', ' Data saved       ')
        else:
            QMessageBox.critical(self, 'Message Box', ' Canceled       ')
            self.savefname = ''

    def savewarp(self):
        # get filename from dialog
        savefname = QFileDialog.getSaveFileName(None, "Select destination folder and file name", ""
                                                , 'csv file(*.csv)')[0]
        if savefname != '':
            bundlenumber = self.spinBox_4.value() - 1
            if self.status_warp == 1:
                output_gam = pd.DataFrame(self.obj.gam)
                t_db = pd.DataFrame({"x": self.time})
                output = pd.concat([t_db, output_gam], axis=1)
                output.to_csv(savefname, index=False)
            elif self.status_warp == 2:
                exec("self.output_gams = pd.DataFrame(self.gams%s)" % (str(bundlenumber)))
                t_db = pd.DataFrame({"x": self.time})
                output = pd.concat([t_db, self.output_gams], axis=1)
                output.to_csv(savefname, index=False)
            QMessageBox.information(self, 'Message Box', ' Data saved       ')
        else:
            QMessageBox.critical(self, 'Message Box', ' Canceled       ')
            self.savefname = ''

    def savesample(self):
        # get filename from dialog
        savefname = QFileDialog.getSaveFileName(None, "Select destination folder and file name", ""
                                                , 'csv file(*.csv)')[0]
        if savefname != '':
            bundlenumber = self.spinBox_4.value() - 1
            t_db = pd.DataFrame({"x": self.time})
            exec("output = pd.concat([t_db, self.df%s], axis=1)" % (str(bundlenumber)))
            exec("output.to_csv(savefname, index=False)")
            QMessageBox.information(self, 'Message Box', ' Sample data saved       ')
        else:
            QMessageBox.critical(self, 'Message Box', ' Canceled       ')
            self.savefname = ''

    def saveUQ(self):
        # get filename from dialog
        savefname = QFileDialog.getSaveFileName(None, "Select destination folder and file name", ""
                                                , 'csv file(*.csv)')[0]
        if savefname != '':
            fsplit = savefname.split('.')
            fext = fsplit[-1]
            fname = fsplit[0:-1]
            output_time = pd.DataFrame({"x": self.time})
            output_upr_cb = pd.DataFrame({"upr_cb": self.upr_cb})
            output_lwr_cb = pd.DataFrame({"lwr_cb": self.lwr_cb})
            output_upr_tb = pd.DataFrame({"upr_tb": self.upr_tb})
            output_lwr_tb = pd.DataFrame({"lwr_tb": self.lwr_tb})
            output_WRS_mean = pd.DataFrame({"WRS_mean": self.final_mean})
            output = pd.concat([output_time, output_upr_cb, output_lwr_cb, output_upr_tb, output_lwr_tb,
                                output_WRS_mean, self.tot_bdmean], axis=1)
            output.to_csv(fname[0] + '.' + fext, index=False)
            QMessageBox.information(self, 'Message Box', ' Data saved       ')
        else:
            QMessageBox.critical(self, 'Message Box', ' Canceled       ')
            self.savefname = ''

    def warping(self):
        self.datasize = self.spinBox.value()
        self.extimetable = range(0, self.datasize)
        self.f_pd = pd.DataFrame()
        for i in range(0, int(self.db_np.shape[1] / 2)):
            i = i + 1
            # inherit valuables
            # x1 =self.x1
            exec("x%s = self.x%s" % (str(i), str(i)))
            # x1 =self.x1
            exec("y%s = self.y%s" % (str(i), str(i)))

            # expand x range according to datasize
            exec("%s = np.array(list(map(int, %s.to_numpy()*%s)))" % ('x' + str(i), 'x' + str(i), 'self.datasize'))

            # delete duplicates
            # x1_1, c1 = np.unique(x1, return_counts=True)
            exec("x%s_1, c%s = np.unique(x%s, return_counts=True)" % (str(i), str(i), str(i)))
            xidx = 0
            idx = []
            exec("self.cshape = c%s.shape[0]" % (str(i)))
            for j in range(0, self.cshape):
                exec("self.append = list(range(xidx, xidx + c%s[j] - 1))" % (str(i)))
                idx.append(self.append)
                exec("xidx = xidx + c%s[j]" % str(i))
            idx = sum(idx, [])
            # y1 = np.delete(y1, idx)
            exec("y%s = np.delete(y%s, idx)" % (str(i), str(i)))

            # make dataframe
            # df1_1 = pd.DataFrame({'y1' : [0] * datasize}, index = self.extimetable)
            exec("df%s_1 = pd.DataFrame({'y%s' : [0] * self.datasize}, index = self.extimetable)" % (str(i), str(i)))
            # df1_2 = pd.DataFrame({'y1' : y1}, index = x1_1)
            exec("df%s_2 = pd.DataFrame({'y%s' : y%s}, index = x%s_1)" % (str(i), str(i), str(i), str(i)))
            # df1_3 = df1_2.combine_first(df1_1)
            exec("df%s_3 = df%s_2.combine_first(df%s_1)" % (str(i), str(i), str(i)))

            # replace 0 with Nan to interpolate
            # df1_4 = df1_3.replace(0, np.NaN)
            exec("df%s_4 = df%s_3.replace(0, np.NaN)" % (str(i), str(i)))

            # interpolation
            # df1_5 = df1_4.interpolate(method='polynomial', order=3, limit_direction = 'both')
            exec("self.df%s_5 = df%s_4.interpolate(method='polynomial', order=3, limit_direction = 'both')" % (str(i), str(i)))
            # unify pre-processed functions
            exec("self.f_pd = pd.concat([self.f_pd, self.df%s_5], axis=1)" % (str(i)))

        self.f_pd = self.f_pd.dropna()

        # valuables to make obj
        self.f = self.f_pd.to_numpy()
        self.time = np.array(self.f_pd.index / self.datasize)

        # obj : standard object
        self.obj = fs.fdawarp(self.f, self.time)
        self.obj.srsf_align(method="median", parallel=False)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self.time, self.obj.fn)
        ax.grid()
        ax.set_xlabel("Normalized Depth from ID (r/t)")
        ax.set_ylabel("Residual Stress")
        ax.set_title("Aligned Functions")
        self.canvas.draw()

        # -load- v
        self.btn_warp.setEnabled(True)
        self.spinBox.setEnabled(True)
        self.exportlabel.setEnabled(True)
        self.btn_savefig.setEnabled(True)
        # ----------------------------------------
        # -warping- v
        self.graphlabel.setEnabled(True)
        self.btn_sample.setEnabled(True)
        self.spinBox_2.setEnabled(True)
        self.spinBox_3.setEnabled(True)
        self.btn_warpshow.setEnabled(True)
        self.chkbox_Kmean.setEnabled(True)
        self.btn_savewarp.setEnabled(True)
        # ----------------------------------------
        # -Sampling- x
        self.btn_UQ.setDisabled(True)
        self.btn_savesample.setDisabled(True)
        # ----------------------------------------
        # -UQ- x
        self.btn_cal.setDisabled(True)
        self.chkbox_mean.setChecked(False)
        self.chkbox_mean.setDisabled(True)
        self.chkbox_cb.setChecked(False)
        self.chkbox_cb.setDisabled(True)
        self.chkbox_tb.setChecked(False)
        self.chkbox_tb.setDisabled(True)
        self.chkbox_WRSmean.setDisabled(True)
        self.btn_saveUQ.setDisabled(True)

        # bundle select
        self.btn_bundleselect.setDisabled(True)
        self.spinBox_4.setDisabled(True)

        self.status_warp = 1

    def sampling(self):
        meanplot = pd.DataFrame([])
        self.bundle_number = self.spinBox_2.value()
        self.sample_number = self.spinBox_3.value()
        for k in range(0, self.bundle_number):
            self.obj.joint_gauss_model(n=self.sample_number)
            exec("self.df%s = pd.DataFrame(self.obj.ft)" % (str(k)))
            exec("self.ft%s = self.obj.ft" % (str(k)))
            exec("self.fn%s = self.obj.fn" % (str(k)))
            exec("self.db%s = self.df%s.to_numpy()" % (str(k), str(k)))
            exec("self.gams%s = self.obj.gams" % (str(k)))
            exec("self.mean = self.db%s.sum(axis=1)/self.sample_number" % (str(k)))
            meandb = pd.DataFrame(self.mean)
            meanplot = pd.concat([meanplot, meandb], axis=1)
            print("processing : " + str(k + 1) + " / " + str(self.bundle_number))

        meanplot1 = meanplot.to_numpy()
        meangap = [0] * meanplot1.shape[0]
        for s in range(0, meanplot1.shape[0]):
            meangap[s] = max(meanplot1[s]) - min(meanplot1[s])

        self.textBrowser.append("gap max = " + str(np.round(max(meangap), 2)) + ", min = " + str(np.round(min(meangap), 2)))

        # -load- v
        self.btn_warp.setEnabled(True)
        self.spinBox.setEnabled(True)
        self.exportlabel.setEnabled(True)
        self.btn_savefig.setEnabled(True)
        # ----------------------------------------
        # -warping- v
        self.graphlabel.setEnabled(True)
        self.btn_sample.setEnabled(True)
        self.spinBox_2.setEnabled(True)
        self.spinBox_3.setEnabled(True)
        # K mean disabled
        self.chkbox_Kmean.setChecked(False)
        self.chkbox_Kmean.setDisabled(True)

        self.btn_savewarp.setEnabled(True)
        self.btn_warpshow.setEnabled(True)
        # ----------------------------------------
        # -Sampling- v
        self.btn_UQ.setEnabled(True)
        self.btn_savesample.setEnabled(True)
        # ----------------------------------------
        # -UQ- x
        self.btn_cal.setDisabled(True)
        self.chkbox_mean.setChecked(False)
        self.chkbox_mean.setDisabled(True)
        self.chkbox_cb.setChecked(False)
        self.chkbox_cb.setDisabled(True)
        self.chkbox_tb.setChecked(False)
        self.chkbox_tb.setDisabled(True)
        self.chkbox_WRSmean.setDisabled(True)
        self.btn_saveUQ.setDisabled(True)

        # bundle select
        self.btn_bundleselect.setEnabled(True)
        self.spinBox_4.setEnabled(True)
        self.spinBox_4.setMaximum(self.spinBox_2.value())
        self.spinBox_4.setValue(self.spinBox_2.value())

        self.status_warp = 2

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        x = self.obj.ft.shape[0]
        ax.plot(self.time, self.obj.ft)
        ax.grid()
        ax.set_xlabel("Normalized Depth from ID (r/t)")
        ax.set_ylabel("Residual Stress")
        ax.set_title("sampled data graph")
        ax.set_xlim([-0.01, 1.01])

        self.canvas.draw()


    def UQ(self):
        N = self.bundle_number
        sampleN = self.sample_number
        self.tot_bdmean = pd.DataFrame([])
        output_Q1 = pd.DataFrame([])
        output_Q3 = pd.DataFrame([])
        op_inv = [0] * N
        upr_tbs = pd.DataFrame([])
        lwr_tbs = pd.DataFrame([])
        for i in range(0, N):
            exec("self.dbn = self.db%s" % (str(i)))
            exec("self.df = self.df%s" % (str(i)))
            bundle_mean = self.dbn.sum(axis=1) / sampleN
            exec("self.bundle_mean_pd = pd.DataFrame({'mean_func%s' : bundle_mean})" % (str(i)))
            dbsum = self.df.sum()
            temp = dbsum.argsort()
            ranks = sampleN - temp.argsort()
            if sampleN > 20:
                Q1_idx = np.where(ranks == int(np.round(sampleN / 100 * 2.5)))
                Q3_idx = np.where(ranks == int(np.round(sampleN / 100 * 97.5)))
            else:
                Q1_idx = np.where(ranks == 1)
                Q3_idx = np.where(ranks == sampleN)
            op1 = self.dbn[:, Q1_idx[0][0]]
            op2 = self.dbn[:, Q3_idx[0][0]]
            op11 = pd.DataFrame(op1)
            op22 = pd.DataFrame(op2)
            self.tot_bdmean = pd.concat([self.tot_bdmean, self.bundle_mean_pd], axis=1)
            output_Q1 = pd.concat([output_Q1, op11], axis=1)
            output_Q3 = pd.concat([output_Q3, op22], axis=1)
            inv = op1.sum() - op2.sum()
            op_inv[i] = inv

            # tolerance bound (Bundle step)
            fp = np.zeros(self.df.shape[0])
            fm = np.zeros(self.df.shape[0])
            for j in range(0, self.dbn.shape[0]):
                amp_list = self.dbn[j]
                s_mean = np.mean(amp_list)
                s_std = np.std(amp_list)
                fp[j] = s_mean + 1.96 * s_std
                fm[j] = s_mean - 1.96 * s_std
            fppd = pd.DataFrame(fp)
            fmpd = pd.DataFrame(fm)
            upr_tbs = pd.concat([upr_tbs, fppd], axis=1)
            lwr_tbs = pd.concat([lwr_tbs, fmpd], axis=1)

        tot_bundlemean = self.tot_bdmean.to_numpy()
        self.upr_cb = np.zeros(self.df.shape[0])
        self.lwr_cb = np.zeros(self.df.shape[0])
        upr_tbs_np = upr_tbs.to_numpy()
        lwr_tbs_np = lwr_tbs.to_numpy()
        self.lwr_tb = np.zeros(self.df.shape[0])
        self.upr_tb = np.zeros(self.df.shape[0])
        self.tb_gap = np.zeros(self.df.shape[0])
        self.cb_gap = np.zeros(self.df.shape[0])
        self.final_mean = np.zeros(self.df.shape[0])

        for i in range(0, self.df.shape[0]):
            # confidence bound
            self.upr_cb[i] = max(tot_bundlemean[i])
            self.lwr_cb[i] = min(tot_bundlemean[i])
            self.cb_gap[i] = self.upr_cb[i] - self.lwr_cb[i]
            self.final_mean[i] = np.mean(tot_bundlemean[i])
            # tolerance bound
            self.upr_tb[i] = max(upr_tbs_np[i])
            self.lwr_tb[i] = min(lwr_tbs_np[i])
            self.tb_gap[i] = self.upr_tb[i] - self.lwr_tb[i]

        cb_gap_mean = np.round(np.mean(self.cb_gap),2)
        tb_gap_mean = np.round(np.mean(self.tb_gap),2)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self.time, self.tot_bdmean)
        ax.plot(self.time, self.upr_tb, 'r--')
        ax.plot(self.time, self.lwr_tb, 'b--')
        ax.plot(self.time, self.upr_cb, 'k')
        ax.plot(self.time, self.lwr_cb, 'k')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([min(self.lwr_tb)*1.05, max(self.upr_tb)*1.05])
        ax.grid()
        ax.set_xlabel("Normalized Depth from ID (r/t)")
        ax.set_ylabel("Residual Stress")
        ax.set_title("Uncertainty Quantification")
        self.canvas.draw()
        self.textBrowser.append("cb gap mean = " + str(cb_gap_mean))
        self.textBrowser.append("tb gap mean = " + str(tb_gap_mean))

        # -load- v
        self.btn_warp.setEnabled(True)
        self.spinBox.setEnabled(True)
        self.exportlabel.setEnabled(True)
        self.btn_savefig.setEnabled(True)
        # ----------------------------------------
        # -warping- v
        self.graphlabel.setEnabled(True)
        self.btn_sample.setEnabled(True)
        self.spinBox_2.setEnabled(True)
        self.spinBox_3.setEnabled(True)

        # warpshow disabled
        self.btn_warpshow.setDisabled(True)
        # K mean disabled
        self.chkbox_Kmean.setChecked(False)
        self.chkbox_Kmean.setDisabled(True)

        self.btn_savewarp.setEnabled(True)
        # ----------------------------------------
        # -Sampling- v
        self.btn_UQ.setEnabled(True)
        self.btn_savesample.setEnabled(True)
        # ----------------------------------------
        # -UQ- v
        self.btn_cal.setEnabled(True)
        self.chkbox_mean.setChecked(True)
        self.chkbox_mean.setEnabled(True)
        self.chkbox_cb.setChecked(True)
        self.chkbox_cb.setEnabled(True)
        self.chkbox_tb.setChecked(True)
        self.chkbox_tb.setEnabled(True)
        self.chkbox_WRSmean.setEnabled(True)
        self.btn_saveUQ.setEnabled(True)

        # bundle select
        self.btn_bundleselect.setDisabled(True)
        self.spinBox_4.setDisabled(True)

    def calculate(self):
        fname2 = QFileDialog.getOpenFileName(self, '', '', 'Excel(*.xlsx *xls)')
        # try:
        dbcom = pd.read_excel(fname2[0])
        self.comp_db_np = dbcom.to_numpy()
        N = int(self.comp_db_np.shape[1] / 2)
        self.f_conpd = pd.DataFrame([])
        for i in range(0, N):
            i = i + 1
            # x1 = db["1(depth)"]
            globals()['comx' + str(i)] = dbcom[str(i) + "x"]
            # y1 = db["1(amp)"]
            globals()['comy' + str(i)] = dbcom[str(i) + "y"]
            # x1 = x1.dropna()
            exec("x%s = comx%s.dropna()" % (str(i), str(i)))
            # y1 = y1.dropna()
            exec("y%s = comy%s.dropna()" % (str(i), str(i)))
            exec("x%s_0 = np.array(list(map(int, x%s.to_numpy()*%s)))" % (str(i), str(i), 'self.datasize'))
            # --- delete duplicates ---
            # x1_1, c1 = np.unique(x1, return_counts=True)
            exec("x%s_1, c%s = np.unique(x%s_0, return_counts=True)" % (str(i), str(i), str(i)))
            xidx = 0
            idx = []
            exec("self.cshape2 = c%s.shape[0]" % (str(i)))
            for j in range(0, self.cshape2):
                exec("self.append2 = list(range(xidx, xidx + c%s[j] - 1))" % (str(i)))
                idx.append(self.append2)
                exec("xidx = xidx + c%s[j]" % str(i))
            idx = sum(idx, [])
            # y1 = np.delete(y1, idx)
            exec("y%s = np.delete(y%s, idx)" % (str(i), str(i)))
            # df1_1 = pd.DataFrame({'y1' : [0] * datasize}, index = self.extimetable)
            exec("df%s_1 = pd.DataFrame({'y%s' : [0] * self.datasize}, index = self.extimetable)" % (str(i), str(i)))
            # df1_2 = pd.DataFrame({'y1' : y1}, index = x1_1)
            exec("df%s_2 = pd.DataFrame({'y%s' : y%s}, index = x%s_1)" % (str(i), str(i), str(i), str(i)))
            # df1_3 = df1_2.combine_first(df1_1)
            exec("df%s_3 = df%s_2.combine_first(df%s_1)" % (str(i), str(i), str(i)))
            # df1_4 = df1_3.replace(0, np.NaN)
            exec("df%s_4 = df%s_3.replace(0, np.NaN)" % (str(i), str(i)))
            # df1_5 = df1_4.interpolate(method='polynomial', order=3, limit_direction = 'both')
            exec("self.com%s = df%s_4.interpolate(method='polynomial', order=3, limit_direction = 'both')" % (
            str(i), str(i)))
            exec("self.f_conpd = pd.concat([self.f_conpd, self.com%s], axis=1)" % (str(i)))
        self.f_conpd = self.f_conpd.dropna()
        self.conf = self.f_conpd.to_numpy()
        self.contime = np.array(self.f_conpd.index / self.datasize)
        # except:
        #     self.textBrowser.append("적절하지 않은 입력입니다.\n 양식이 적절한 지 확인해주세요.")

        # Calculate RMSE & Diff
        f_med = self.final_mean
        f = self.conf
        length = f.shape[0]
        diff_1 = np.zeros(length)
        diff_2 = np.zeros(length)
        RMSE = [0] * N
        diff_avg = [0] * N
        len10p = int(np.round(length / 10))
        if len10p == 0:
            len10p = 1

        # Calculate RMSE, diff_avg
        for i in range(0, N):
            for k in range(0, length):
                diff_1[k] = f[k, i] - f_med[k]
                diff_2[k] = diff_1[k] ** 2
            diff_s = np.sum(diff_2)
            RMSE[i] = np.sqrt(diff_s / length)
            diff_avg[i] = np.sum(diff_1[0:(len10p - 1)]) / len10p

        RMSE = np.round(RMSE, 1)
        diff_avg = np.round(diff_avg, 1)
        self.textBrowser.append("        RMSE     diff_avg")
        for i in range(0, N):
            self.textBrowser.append("func" + str(i+1) + " : " + str(RMSE[i]) + "     " + str(diff_avg[i]))
        self.textBrowser.append("end")

    def replot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if self.chkbox_mean.isChecked():
            ax.plot(self.time, self.tot_bdmean)
        if self.chkbox_cb.isChecked():
            ax.plot(self.time, self.upr_cb, 'k')
            ax.plot(self.time, self.lwr_cb, 'k')
        if self.chkbox_tb.isChecked():
            ax.plot(self.time, self.upr_tb, 'r--')
            ax.plot(self.time, self.lwr_tb, 'b--')

        if self.chkbox_WRSmean.isChecked():
            ax.plot(self.time, self.final_mean, color='r', linewidth="3")
        ax.grid()
        ax.set_xlabel("Normalized Depth from ID (r/t)")
        ax.set_ylabel("Residual Stress")
        ax.set_title("Uncertainty Quantification")
        ax.set_xlim([-0.01, 1.01])
        try:
            ax.set_ylim([min(self.lwr_tb)*1.05, max(self.upr_tb)*1.05])
        except:
            print('tolerance bound is not calculated')
        self.canvas.draw()

    def Kplot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self.time, self.obj.fn)
        if self.chkbox_Kmean.isChecked():
            ax.plot(self.time, self.obj.fmean, color='r', linewidth="3")
        ax.grid()
        ax.set_xlabel("Normalized Depth from ID (r/t)")
        ax.set_ylabel("Residual Stress")
        ax.set_title("Aligned Functions")
        self.canvas.draw()

    def warpplot_toggled(self, checked):
        bundlenumber = self.spinBox_4.value() - 1
        if self.btn_warpshow.isChecked():
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            if self.status_warp == 1:
                ax.plot(self.time, self.obj.gam)
            elif self.status_warp == 2:
                exec("ax.plot(self.time, self.gams%s)" % str(bundlenumber))
            ax.grid()
            ax.set_xlabel("Normalized Depth from ID (r/t)")
            ax.set_ylabel("gamma(γ)")
            ax.set_title("Warping Functions")
            # ax.set_xlim([-0.01, 1.01])
            self.canvas.draw()
            self.chkbox_Kmean.setDisabled(True)
        else:
            if self.status_warp == 1:
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                exec("ax.plot(self.time, self.fn%s)" % str(bundlenumber))
                ax.grid()
                ax.set_xlabel("Normalized Depth from ID (r/t)")
                ax.set_ylabel("Residual Stress")
                ax.set_title("Aligned Functions")
                self.canvas.draw()
                self.chkbox_Kmean.setEnabled(True)
                self.chkbox_Kmean.setChecked(False)
            elif self.status_warp == 2:
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                x = self.obj.ft.shape[0]
                exec("ax.plot(self.time, self.ft%s)" % str(bundlenumber))
                ax.grid()
                ax.set_xlabel("Normalized Depth from ID (r/t)")
                ax.set_ylabel("Residual Stress")
                ax.set_title("sampled data graph")
                ax.set_xlim([-0.01, 1.01])
                self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Mainwindow()
    sys.exit(app.exec())