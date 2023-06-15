"""
This code is built for a DCSC BEP project for the bachelor Mechanical Engineering at the TU Delft 2023.
It was used to do synchronchronization research on airfoils in a small windtunnel.
ChatGPT was used to help built some of the code.
"""

import serial
import serial.tools.list_ports
import time
import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from matplotlib import cm
from scipy.interpolate import griddata

class PitchFoilRun:
    def __init__(self, run_name):
        self.run_name = run_name # name of the test run
        self.chord_length = 0.125 # chord lenght of the airfoil, it should be a SD7003 airfoil
        self.test_time = 40.0 # test time in seconds per test
        self.wind_speed = 0.00000000000000000001 # wind speed will be defined later
        self.alpha = 20.0 # Determine oscillation amplitude in degrees
        self.freq = 1.0 # Deterime oscillation frequency in Hz
        if not os.path.exists(os.getcwd()+'\\Runs'): #Creating folders to save results
            os.mkdir(os.getcwd()+'\\Runs')
        if not os.path.exists(os.getcwd()+'\\Runs\\' + self.run_name):
            os.mkdir(os.getcwd()+'\\Runs\\' + self.run_name)
        self.folder_location = os.getcwd()+'\\Runs\\' + self.run_name
        self.open_test_matrix()
        print('open_test_matrix done')
        self.determine_offset()
        print('determine_offset done')
        self.determine_baseline()
        print('determine_baseline done')
        self.open_or_make_results()
        print('open_or_make_results done')
        self.save_settings()
        print('save_settings done')
        self.open_or_make_filtered_results()
        print('open_or_make_filtered_results done')
        print("DONE")
        print('Ready for plotting')

    def open_test_matrix(self):
        """
        This function opens a file named 'test_matrix.xlsx' which should contain the relative test matrix for the test run.
        Also it saves a absolute version and a shorter version of this test matrix.
        """
        if not os.path.exists(self.folder_location+'/relative_test_matrix '+ self.run_name +'.xlsx'):
            self.relative_test_matrix = pd.read_excel('test_matrix.xlsx', index_col=0) # Open test_matrix (with relative values)
            self.relative_test_matrix.to_excel(self.folder_location+'/relative_test_matrix '+ self.run_name +'.xlsx') #Save the test_matrix again in the run folder
        else:
            self.relative_test_matrix = pd.read_excel(self.folder_location+'/relative_test_matrix '+ self.run_name +'.xlsx',index_col=0)

        relative_list = []
        for test_index in self.relative_test_matrix.index:
            relative_list.append(self.relative_test_matrix.loc[test_index])
        relative_list = np.array(relative_list)

        self.short_relative_test_matrix = self.make_short_relative_matrix(self.relative_test_matrix)
        self.number_of_tests = int(len(self.short_relative_test_matrix.index))
        print(self.number_of_tests)

    def determine_offset(self):
        """
        This function makes shure that three tests are done to determine the offset on the airfoils caused by their weight.
        """
        offset_relative_test_list = [[0,0,0],[0,0,0],[0,0,0]]
        offset_relative_test_matrix = pd.DataFrame(offset_relative_test_list, index =  ['test 1.1','test 1.2','test 1.3'] , columns=['relativeA','P','relativef'])
        offset_absolute_test_list = [[0,0,0,1,1],[0,0,0,1,1],[0,0,0,1,1]]
        offset_absolute_test_matrix = pd.DataFrame(offset_absolute_test_list, index =  ['test 1.1','test 1.2','test 1.3'] , columns=['A1','A2','P','f1','f2'])
        
        if not os.path.exists(self.folder_location+'/offset results '+ self.run_name +'.xlsx'):
            print('Put the wind tunnel OFF!!!')
            input('and press ENTER here to continue:')
            offset_results, row_names = self.run_tests(offset_absolute_test_matrix, offset_relative_test_matrix)
            self.offset_results = pd.DataFrame(offset_results, index = row_names)
            self.offset_results.to_excel(self.folder_location+'/offset results '+ self.run_name +'.xlsx')
        else:
            self.offset_results = pd.read_excel(self.folder_location+'/offset results '+ self.run_name +'.xlsx',  index_col=0)

        #Determine the offset from the offset file
        self.S1_offset = (np.average(self.drop_nans(self.offset_results.loc['test 1.1 S1']))+np.average(self.drop_nans(self.offset_results.loc['test 1.2 S1']))+np.average(self.drop_nans(self.offset_results.loc['test 1.3 S1'])))/3
        self.S2_offset = (np.average(self.drop_nans(self.offset_results.loc['test 1.1 S2']))+np.average(self.drop_nans(self.offset_results.loc['test 1.2 S2']))+np.average(self.drop_nans(self.offset_results.loc['test 1.3 S2'])))/3
        self.S3_offset = (np.average(self.drop_nans(self.offset_results.loc['test 1.1 S3']))+np.average(self.drop_nans(self.offset_results.loc['test 1.2 S3']))+np.average(self.drop_nans(self.offset_results.loc['test 1.3 S3'])))/3

    def determine_baseline(self):
        """
        This function makes shure that three tests are done to determine the baseline normal force on the airfoils when the wind tunnel is on and the foils are not actuated
        """
        baseline_relative_test_list = [[0,0,0],[0,0,0],[0,0,0]]
        baseline_relative_test_matrix = pd.DataFrame(baseline_relative_test_list, index =  ['test 1.1','test 1.2','test 1.3'] , columns=['relativeA','P','relativef'])
        baseline_absolute_test_list = [[0,0,0,1,1],[0,0,0,1,1],[0,0,0,1,1]]
        baseline_absolute_test_matrix = pd.DataFrame(baseline_absolute_test_list, index =  ['test 1.1','test 1.2','test 1.3'] , columns=['A1','A2','P','f1','f2'])
        
        if not os.path.exists(self.folder_location+'/baseline results '+ self.run_name +'.xlsx'):
            print('Put the wind tunnel ON!!!')
            self.wind_speed = float(input('What is the wind speed:'))
            input('and press ENTER here to continue:')
            baseline_results, row_names = self.run_tests(baseline_absolute_test_matrix, baseline_relative_test_matrix)
            self.baseline_results = pd.DataFrame(baseline_results, index = row_names)
            self.baseline_results.to_excel(self.folder_location+'/baseline results '+ self.run_name +'.xlsx')
        else:
            self.baseline_results = pd.read_excel(self.folder_location+'/baseline results '+ self.run_name +'.xlsx',  index_col=0)

        self.S1_baseline = (np.average(self.drop_nans(self.baseline_results.loc['test 1.1 S1']))+np.average(self.drop_nans(self.baseline_results.loc['test 1.2 S1']))+np.average(self.drop_nans(self.baseline_results.loc['test 1.3 S1'])))/3
        self.S2_baseline = (np.average(self.drop_nans(self.baseline_results.loc['test 1.1 S2']))+np.average(self.drop_nans(self.baseline_results.loc['test 1.2 S2']))+np.average(self.drop_nans(self.baseline_results.loc['test 1.3 S2'])))/3
        self.S3_baseline = (np.average(self.drop_nans(self.baseline_results.loc['test 1.1 S3']))+np.average(self.drop_nans(self.baseline_results.loc['test 1.2 S3']))+np.average(self.drop_nans(self.baseline_results.loc['test 1.3 S3'])))/3

    def open_or_make_results(self):
        """
        This function looks wether there are already results saved, depending on wether they are already saved it will open them or start running the tests
        """
        if not os.path.exists(self.folder_location+'/results '+ self.run_name +'.xlsx'):
            print('Keep the wind tunnel ON!!!')
            input('and press ENTER here to start testing:')
            results, row_names = self.run_tests(self.absolute_test_matrix, self.relative_test_matrix)
            self.results = pd.DataFrame(results, index = row_names)
            self.results.to_excel(self.folder_location+'/results '+ self.run_name +'.xlsx')
        else:
            self.results = pd.read_excel(self.folder_location+'/results '+ self.run_name +'.xlsx',  index_col=0)

    def save_settings(self):
        """
        This function saves tome of the settings at which the test run was done
        """
        if not os.path.exists(self.folder_location+'/settings '+ self.run_name +'.xlsx'):
            Strou = (self.freq*self.chord_length*0.6132*2*np.sin(self.alpha*(np.pi/180)))/self.wind_speed

            settings = [self.test_time, self.chord_length, self.wind_speed, self.alpha, self.freq, Strou]
            settings_index = ['Test Time (s)', 'Chord Length (m)', 'Wind Speed (m/s)', 'Amplitude (degrees)', 'Frequensy (Hz)', 'Strouhal number']
            self.settings_data = pd.DataFrame(settings, index = settings_index)
            self.settings_data.to_excel(self.folder_location+'/settings '+ self.run_name +'.xlsx')
        else:
            self.settings_data = pd.read_excel(self.folder_location+'/settings '+ self.run_name +'.xlsx',  index_col=0)


    def Rel_to_Abs(self, Rel):
        """
        Function to build test matrix with absolute values from test matrix with relative values
        """
        Abs = []
        for test in Rel:  # Write relative alpha and f to absolute per test
            Rel_alpha = test[0]
            P_degrees = test[1] #phase delay in degrees
            Rel_f = test[2] 

            if Rel_alpha == 0:
                Abs_alpha1 = 0
            else:
                Abs_alpha1 = self.alpha
            Abs_alpha2 = Abs_alpha1*Rel_alpha
            if Rel_f == 0:
                Abs_f1 = 1
                Abs_f2 = 1
            else:
                Abs_f1 = self.freq
                Abs_f2 = Abs_f1*Rel_f
            Abs.append([Abs_alpha1,Abs_alpha2,P_degrees,Abs_f1,Abs_f2])
        return Abs

    def make_short_relative_matrix(self, rel_test_matrix):
        """
        This function creates a shorter functino of the test matrix in which tests which are repeated are only shown ones
        """
        short_rel_list = []
        index_list = []
        for row in rel_test_matrix.index:
            if row.endswith('.1'):
                short_rel_list.append(rel_test_matrix.loc[row])
                index_list.append(row.replace('.1',''))
        if len(short_rel_list) == 0:
            return rel_test_matrix
        else:
            short_rel_frame = pd.DataFrame(short_rel_list, index = index_list)
            return short_rel_frame

    def run_tests(self, abs_test_matrix, rel_test_matrix):
        """
        This function communicates with the Arduino to actuated the airfoils and get back the strain gauge measurements
        """
        #Connect to Serial
        serial_port = "COM4"
        baud_rate = 115200
        arduino_serial = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for the serial connection to initialize

        row_names = []
        for test_index in rel_test_matrix.index:
            row_names.append(test_index +  ' time')
            row_names.append(test_index + ' A1')
            row_names.append(test_index + ' A2')
            row_names.append(test_index + ' S1')
            row_names.append(test_index + ' S2')
            row_names.append(test_index + ' S3')

        #Run the tests
        results = []
        for n,test in enumerate(abs_test_matrix.values.tolist()):
            while True:
                try: # Makes sure the test is tried again when something goes wrong in the communication
                    start_time = time.time()
                    A1, A2, P,  f1, f2 = test[0], test[1], test[2], test[3], test[4]
                    print(A1, A2, P, f1, f2)
                    arduino_serial.write(f"start:{A1},{A2};{P}-{f1}+{f2}\n".encode()) # Send start command with desired setup

                    test_results = []
                    while arduino_serial.in_waiting or (time.time() - start_time < self.test_time):
                        response = arduino_serial.readline().decode("ascii").strip() # Read data comming back from arduino
                        print(response)
                        test_results.append(list(np.float_(response.split(';'))))
                        if time.time() - start_time > self.test_time:
                            arduino_serial.write(f"stop\n".encode()) # Stop last test
                    print(abs_test_matrix.index[n]+ ' done')
                    
                    arduino_serial.reset_input_buffer()
                    results.append(test_results)
                    time.sleep(5)
                    arduino_serial.reset_input_buffer()
                    break
                except Exception:
                    arduino_serial.write(f"stop\n".encode())
                    time.sleep(5)
                    arduino_serial.reset_input_buffer()
                    time.sleep(5)
                    continue
        results_backup = pd.DataFrame(results)
        results_backup.to_excel(self.folder_location+'/results backup '+ self.run_name +'.xlsx') # Saves a non reordered back up of the test results

        results = self.reorder_results(results)

        return results, row_names
    
    def reorder_results(self, results):
        """
        This function reorders the results before they are saved in an excel file
        """
        reordered_results = []
        for test in results:
            reordered_results.append(np.array(test)[:,0])
            reordered_results.append(np.array(test)[:,1])
            reordered_results.append(np.array(test)[:,2])
            reordered_results.append(np.array(test)[:,3])
            reordered_results.append(np.array(test)[:,4])
            reordered_results.append(np.array(test)[:,5])
        return reordered_results
    

    def moving_average(self, x, w):
        """
        This function aplies a moving average filter on list x with batch size w
        """
        window = int(w/2)
        return np.array([np.mean(x[max(0, i-window):min(len(x), i+window+1)]) for i in range(len(x))])
    
    def drop_nans(self, lst):
        """
        Drops the NaN values from a list
        """
        return [x for x in lst if not math.isnan(x)]
    
    def open_or_make_filtered_results(self):
        """
        Creates a file where all results are filtered using the moving average filter
        """
        if not os.path.exists(self.folder_location+'/filtered results '+ self.run_name +'.xlsx'):
            MA_w = 100
            unfiltered_results = self.results
            filtered_results_list = []
            for row in self.results.index:
                if row.endswith('time'):
                    t = unfiltered_results.loc[row].values.tolist()
                if row.endswith('S1') or row.endswith('S2') or row.endswith('S3'):
                    filtered_row = self.moving_average(unfiltered_results.loc[row].values.tolist(),MA_w)
                    filtered_results_list.append(filtered_row)
                else:
                    filtered_results_list.append(unfiltered_results.loc[row])
            self.filtered_results = pd.DataFrame(filtered_results_list, index = self.results.index)
            self.filtered_results.to_excel(self.folder_location+'/filtered results '+ self.run_name +'.xlsx')
        else:
            self.filtered_results = pd.read_excel(self.folder_location+'/filtered results '+ self.run_name +'.xlsx',  index_col=0)

    def is_even(self, number):
        """
        Test wether the variable 'number' is a even and returns either True or False 
        """
        if number % 2 == 0:
            return True
        else:
            return False
   
    def plot_signal(self):
        """
        Function to make a time plot of all signals
        """
        van = 1515
        tot = 3030  
        for test_index in self.relative_test_matrix.index:
            fig, ax1 = plt.subplots(layout='constrained', figsize=(5.5, 4))
            ax2 = ax1.twinx()
            for row in self.results.index:
                if row.startswith(test_index+' '):
                    if row.endswith('S1'):
                        
                        ax2.plot(self.filtered_results.loc[test_index + ' time'][van:tot]/1000, (self.filtered_results.loc[row][van:tot]-self.S1_offset)*(5/1024),color='red',label='S1 [V]')
                        continue
                    elif row.endswith('S2'):
                        ax2.plot(self.filtered_results.loc[test_index + ' time'][van:tot]/1000, (self.filtered_results.loc[row][van:tot]-self.S2_offset)*(5/1024),color='darkorange',label='S2 [V]')
                        continue
                    elif row.endswith('S3'):
                        ax2.plot(self.filtered_results.loc[test_index + ' time'][van:tot]/1000, (self.filtered_results.loc[row][van:tot]-self.S3_offset)*(5/1024),color='gold',label='S3 [V]')
                        continue
                    elif row.endswith('A1'):
                        ax1.plot(self.filtered_results.loc[test_index + ' time'][van:tot]/1000, self.filtered_results.loc[row][van:tot],color='deepskyblue', dashes=[6,2], label='A1 [deg]')
                        continue
                    elif row.endswith('A2'):
                        ax1.plot(self.filtered_results.loc[test_index + ' time'][van:tot]/1000, self.filtered_results.loc[row][van:tot],color='royalblue', dashes=[5,2],label='A2 [deg]')
                        continue
                    else:
                        continue
                    
            ax1.grid(which='both',axis='both')
            ax2.grid(which='both',axis='both')
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Angle [deg]')
            ax2.set_ylabel('Arduino Input (minus offset) [V]')
            ax2.set_ylim(-0.7,0.7)
            fig.legend(loc='outside upper center', mode = "expand", ncol = 5)
            fig.savefig(self.folder_location + '/Signal Plot '+ test_index + ' ' + self.run_name + '.svg', bbox_inches='tight')
            plt.close()
    
    def plot_PNCombi(self):
        """
        Function to plot phase delay against normal force proxy for airfoil
        """
        F1 = []
        F2 = []
        F3 = []
        F23 = []

        for n,test_index in enumerate(self.short_relative_test_matrix.index):
            if self.short_relative_test_matrix.loc[test_index, 'relativef'] == 1 and self.short_relative_test_matrix.loc[test_index, 'relativeA'] == 1:
                Nforce23 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S2']))-self.S2_offset+np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S3']))-self.S3_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S2']))-self.S2_offset+np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S3']))-self.S3_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S2']))-self.S2_offset+np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S3']))-self.S3_offset))/3
                Nforce1 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S1']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S1']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S1']))-self.S1_offset))/3
                Nforce2 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S2']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S2']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S2']))-self.S1_offset))/3
                Nforce3 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S3']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S3']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S3']))-self.S1_offset))/3

                F1.append([self.short_relative_test_matrix.loc[test_index, 'P'], Nforce1])
                F2.append([self.short_relative_test_matrix.loc[test_index, 'P'], Nforce2])
                F3.append([self.short_relative_test_matrix.loc[test_index, 'P'], Nforce3])
                F23.append([self.short_relative_test_matrix.loc[test_index, 'P'], Nforce23])

        Baseforce23 = abs(self.S2_baseline + self.S3_baseline - self.S2_offset - self.S3_offset)
        Baseforce1 = abs(self.S1_baseline - self.S1_offset)
        Baseforce2 = abs(self.S2_baseline - self.S2_offset)
        Baseforce3 = abs(self.S3_baseline - self.S3_offset)

        fig, ax = plt.subplots(layout='constrained', figsize=(5.5, 4))
        colors = ['red','darkorange','gold', 'green']
        labels = ['F1','F2','F3','F2+F3']
        BaseF = [Baseforce1,Baseforce2,Baseforce3,Baseforce23]
        for n, DATA in enumerate([F1, F2, F3, F23]):
            DATA = np.array(DATA)
            x = DATA[:,0]
            y = DATA[:,1]
            y = y/BaseF[n]
            data = list(zip(x,y))
            sorted_data = sorted(data, key=lambda pair: pair[0])
            x, y = zip(*sorted_data)        
            ax.scatter(x, y, color=colors[n], label=labels[n])  # Add scatter plot for data points
            ax.plot(x,y,color=colors[n])

        ax.grid(which='both',axis='both')
        ax.set_xlabel('Phase [deg]')
        ax.set_ylim(-0.5,2)
        ax.set_ylabel('Normal Force Proxy (normalized)')
        fig.legend(loc='outside upper center', mode = "expand", ncol = 4)
        fig.savefig(self.folder_location + '/PN Combi ' + self.run_name + '.svg', bbox_inches='tight')
        plt.show()

    def plot_FNCombi(self):
        """
        Function to create a relative frequency vs normal force proxy plot, which shows a line for each foil
        """
        F1 = []
        F2 = []
        F3 = []
        F23 = []

        for n,test_index in enumerate(self.short_relative_test_matrix.index):
            if self.short_relative_test_matrix.loc[test_index, 'relativeA'] == 1 and self.short_relative_test_matrix.loc[test_index, 'P'] == 0:
                Nforce23 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S2']))-self.S2_offset+np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S3']))-self.S3_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S2']))-self.S2_offset+np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S3']))-self.S3_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S2']))-self.S2_offset+np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S3']))-self.S3_offset))/3
                Nforce1 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S1']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S1']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S1']))-self.S1_offset))/3
                Nforce2 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S2']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S2']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S2']))-self.S1_offset))/3
                Nforce3 = ((np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.1 S3']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.2 S3']))-self.S1_offset)+(np.average(self.drop_nans(self.results.loc[self.short_relative_test_matrix.index[n] + '.3 S3']))-self.S1_offset))/3

                F1.append([self.short_relative_test_matrix.loc[test_index, 'relativef'], Nforce1])
                F2.append([self.short_relative_test_matrix.loc[test_index, 'relativef'], Nforce2])
                F3.append([self.short_relative_test_matrix.loc[test_index, 'relativef'], Nforce3])
                F23.append([self.short_relative_test_matrix.loc[test_index, 'relativef'], Nforce23])

        Baseforce23 = abs(self.S2_baseline + self.S3_baseline - self.S2_offset - self.S3_offset)
        Baseforce1 = abs(self.S1_baseline - self.S1_offset)
        Baseforce2 = abs(self.S2_baseline - self.S2_offset)
        Baseforce3 = abs(self.S3_baseline - self.S3_offset)

        fig, ax = plt.subplots(layout='constrained', figsize=(5.5, 4))
        colors = ['red','darkorange','gold', 'green']
        labels = ['F1','F2','F3','F2+F3']
        BaseF = [Baseforce1,Baseforce2,Baseforce3,Baseforce23]
        for n, DATA in enumerate([F1, F2, F3, F23]):
            DATA = np.array(DATA)
            x = DATA[:,0]
            y = DATA[:,1]
            y = y/BaseF[n]
            data = list(zip(x,y))
            sorted_data = sorted(data, key=lambda pair: pair[0])
            x, y = zip(*sorted_data)        
            ax.scatter(x, y, color=colors[n], label=labels[n]) 
            ax.plot(x,y,color=colors[n])

        ax.grid(which='both',axis='both')
        ax.set_xlabel('Relative Frequency')
        ax.set_ylabel('Normal Force Proxy (normalized)')
        fig.legend(loc='outside upper center', mode = "expand", ncol = 4)
        fig.savefig(self.folder_location + '/FN Combi ' + self.run_name + '.svg', bbox_inches='tight')
        plt.show()



if __name__ == "__main__":
    Run = PitchFoilRun('RUN4')
    Run.plot_signal()

    Run.plot_FNCombi()




