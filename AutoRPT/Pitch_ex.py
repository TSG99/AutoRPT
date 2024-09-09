#!/usr/bin/env python
# coding: utf-8

# In[1]:


import parselmouth
import tgt
import numpy as np

class PitchExtraction:
    
    def getMaxPitch(self, Wav_file, start_time, end_time):
        sub_sound = Wav_file.extract_part(from_time=start_time, to_time=end_time)
        pitch = sub_sound.to_pitch()
        max_pitch = max(pitch.selected_array['frequency'])
        return max_pitch

    #Grabbing min pitch for interval
    def getMinPitch(self, Wav_file, start_time, end_time):
        sub_sound = Wav_file.extract_part(from_time = start_time, to_time = end_time)
        pitch = sub_sound.to_pitch()
        min_pitch = min(pitch.selected_array['frequency'])
        return min_pitch

    #Calculate Standard Deviation
    def getPitchStandardDeviation(self, Wav_file, start_time, end_time):
        sub_sound = Wav_file.extract_part(from_time=start_time, to_time=end_time)
        pitch = sub_sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_std_dev = np.std(pitch_values)
        return pitch_std_dev

    #Calculate average of pitch in interval
    def getAveragePitch(self, Wav_file, start_time, end_time):
        sub_sound = Wav_file.extract_part(from_time=start_time, to_time=end_time)
        pitch = sub_sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        average_pitch = sum(pitch_values) / len(pitch_values)
        return average_pitch


# In[2]:


import parselmouth
import tgt
import numpy as np


class SpeakerNormalization:

    #Takes arr and returns the average of the values
    def fileMean(self, interval_data, arr):
        mean_sum = 0
        mean_n = len(interval_data[arr])
        sqr_diff = 0
        for value in interval_data[arr]:
            mean_sum += value
        file_avg = mean_sum / mean_n
        
        return file_avg
    
    #Takes arr and average and returns Standard Deviation (Std) of the values
    def fileStd(self, interval_data, avg, arr):
        
        mean_n = len(interval_data[arr])
        
        sqr_diff = 0
        
        for value in interval_data[arr]:
            
            sqr_diff += (value - avg) * (value - avg)
    
        sqr_mean = sqr_diff / mean_n
    
        file_std = sqr_mean ** 0.5
        
        return file_std
    
    #Takes arr and returns the minimum value
    def fileMin(self, interval_data, arr):
    
        file_min = min(interval_data[arr])
        
        return file_min
    
    #Takes arr and returns the maximum value
    def fileMax(self, interval_data, arr):
    
        file_max = max(interval_data[arr])
    
        return file_max
    
    #Takes arr, average, Std and appends the Z-score to dict
    def zScoreAppend(self, interval_data, avg, std, arr):
    
        for value in interval_data[arr]:
        
            z_score = (value - avg) / std
        
            interval_data["z-score"].append(z_score)
    
        return interval_data
    
    def getZScore(self, key, avg, std):
        z_score = (key - avg) / std
        return z_score


# In[3]:


import parselmouth
import tgt
import numpy as np

class FileProcessor:
    
    def __init__(self):
        self.pe = PitchExtraction()

    
    def iterateTextGridforPitch(self, Textgrid_path, tier_name, Wav_file):
    
        error_count = 0
        error_arr = []
        #Create Dictionary
        interval_data = {"Interval":[],"Text":[], "min":[], "max":[], "mean":[], "Std":[], "z-score":[], "start":[], "end":[], "STD":[], "Z-SCORE":[]}
    
        #Load the TextGrid using tgt
        tgt_text_grid = tgt.io.read_textgrid(Textgrid_path)
    
        average_sum = 0
        count = 0
        dict_iterable = 0

        #Get the specified tier
        tier = None
        for t in tgt_text_grid.tiers:
            if t.name == tier_name:
                tier = t
                break

        if tier is None:
            print(f"Tier '{tier_name}' not found in the TextGrid.")
            return

        #Iterate through intervals on the tier
        for interval in tier:
            start_time = interval.start_time
            end_time = interval.end_time
            interval_text = interval.text
            #print("start time:", start_time, ", end time:", end_time, ", text:", interval_text)
        
            if interval_text[0] == "{":
                pass
            else:
                
                try:
                    
                    pitch_std_dev = self.pe.getPitchStandardDeviation(Wav_file, start_time, end_time)
        
                    interval_data["Std"].append(pitch_std_dev)
                    
                    interval_data["Text"].append(interval_text)
        
                    dict_iterable += 1
        
                    interval_data["Interval"].append(dict_iterable)
            
                    interval_data["start"].append(start_time)
                
                    interval_data["end"].append(end_time)
        
                    #Calculate the pitch standard deviation for the interval
        
                    #Calculate Max pitch of interval
                    high = self.pe.getMaxPitch(Wav_file, start_time, end_time)
        
                    interval_data["max"].append(high)
        
                    #Calculate Min pitch of interval
                    low = self.pe.getMinPitch(Wav_file, start_time, end_time)
        
                    interval_data["min"].append(low)
            
                    #get the average pitch of interval
                    average = self.pe.getAveragePitch(Wav_file, start_time, end_time)
        
                    interval_data["mean"].append(average)
        
            #Find the average of all intervals so far
                    average_sum += average
                    count += 1
                    total_average = average_sum / count
                
                except Exception as e:
                    #print(f"Skipping Interval due to error: {e}")
                    error_count = error_count + 1
                    dict_iterable +=1
                    error_arr.append(dict_iterable)
        
            #Predict Prosidic event
            #if high > (average + (pitch_std_dev * 2)):
            #    P = "Yes"
           # else:
           #     P = "No"
        
           # print("Pitch Summary:")
           # print("Low:", low, "|High:", high, "|Average:", average, "|Standard Deviation:", pitch_std_dev )
           # print("The total average pitch so far is: ", total_average)
           # print("Prosodic Event Prediction:", P)
           # print("\n")
        
       # print(interval_data)
        
        return interval_data, error_count, error_arr


# In[ ]:





# In[4]:


class FormatToInterval:
    
    def dictToArr(self, arr):
        
        tier_arrays = []
        
        # Initialize first array with formatting
        header = ["Interval", "Text", "Min", "Max", "Mean", "Standard Deviation", "Z-Score", "Start", "End"]
        tier_arrays.append(header)
        
        # Create User-Output Array
        for i in range(len(arr["Interval"])):
            row = [
                arr["Interval"][i],
                arr["Text"][i],
                arr["min"][i],
                arr["max"][i],
                arr["mean"][i],
                arr["STD"][i],
                arr["Z-SCORE"][i],
                arr["start"][i],
                arr["end"][i]
            ]
            tier_arrays.append(row)
            
        return tier_arrays

    def outputArr(self, arr):
        
        for i, array in enumerate(arr):
            print(array)


# In[ ]:





# In[5]:


import csv

class Formatting():
    def to_csv(self, data, csv_file):
        # Specify the CSV file name
        
        
        
        # Write the sub-arrays to the CSV file
        with open(csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
    
            # Iterate through the sub-arrays and write each element in separate columns
            for sub_array in data:
                 csv_writer.writerow(sub_array)

        print(f'Data has been written to {csv_file}.')


# In[ ]:





# In[6]:


import csv

class Training:

    def compare_x(self, csv_file, x_values, result_column):
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            fieldnames = csv_reader.fieldnames + [result_column]
            rows = []
            for row in csv_reader:
                row_values = {key: float(value) for key, value in row.items() if key in ['Start', 'End']}
                row[result_column] = 0  # Default value for the result column
                for x in x_values:
                    for key, value in row_values.items():
                        if value < x <= row_values['End']:
                            row[result_column] = 1
                            break  # Break the loop if condition met for any x
                rows.append(row)

        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def main(self, ton_file, csv_file):

        # Extract timestamps associated with '*' and '%'
        star_timestamps = []
        percent_timestamps = []
        with open(ton_file, 'r') as file:
            header_found = False
            for line in file:
                if line.startswith('#'):
                    header_found = True
                    continue  # Skip the header line
                if not header_found:
                    continue  # Skip lines before the header
                if line.strip().endswith('*'):
                    star_timestamps.append(float(line.split()[0]))  # Assuming the timestamp is the first token in the line
                elif line.strip().endswith('%'):
                    percent_timestamps.append(float(line.split()[0]))  # Assuming the timestamp is the first token in the line

        print("Star Timestamps:", star_timestamps)
        print("Percent Timestamps:", percent_timestamps)

        # Compare timestamps with CSV data
        star_result_column = 'Prom_Star'
        percent_result_column = 'Prom_Percent'
        self.compare_x(csv_file, star_timestamps, star_result_column)
        self.compare_x(csv_file, percent_timestamps, percent_result_column)


# In[ ]:





# In[7]:


class plswrk:
    
    def contextWindow(self, complete_data):
    
        f = -3
        g = -2
        h = -1
        i= 0
        j= 1
        k= 2
        l = 3

        for element in complete_data['max']:
    
            #After only
            if i == 0:
                num = [complete_data['max'][i], complete_data['max'][j], complete_data['max'][k], complete_data['max'][l]]
            
            #1 Before
            elif i == 1:
                num = [complete_data['max'][h], complete_data['max'][i], complete_data['max'][j], complete_data['max'][k], complete_data['max'][l]]
        
            #2 before
            elif i == 2:
                num = [complete_data['max'][g], complete_data['max'][h], complete_data['max'][i], complete_data['max'][j], complete_data['max'][k], complete_data['max'][l]]
                  
##################################################################
        
            #2 after
            elif i == (len(complete_data['max'])- 3):
                num = [complete_data['max'][f], complete_data['max'][g], complete_data['max'][h], complete_data['max'][i], complete_data['max'][j], complete_data['max'][k]]
    
            #1 after
            elif i == (len(complete_data['max']) -2):
                num = [complete_data['max'][f], complete_data['max'][g], complete_data['max'][h], complete_data['max'][i], complete_data['max'][j]]
        
            #Before only
            elif i == (len(complete_data['max']))-1:
                num = [complete_data['max'][f], complete_data['max'][g], complete_data['max'][h], complete_data['max'][i]]
            
########################################################
            #All other cases
            else:
                num = [complete_data['max'][f], complete_data['max'][g], complete_data['max'][h], complete_data['max'][i], complete_data['max'][j], complete_data['max'][k], complete_data['max'][l]]
            
            
            std = np.std(num)
            avg = np.mean(num)
            z_score = (complete_data['max'][i] - avg) / std            
            complete_data['STD'].append(std)
            complete_data['Z-SCORE'].append(z_score)
        
            f += 1
            g += 1
            h += 1
            i += 1
            j += 1
            k += 1
            l += 1
            
        return complete_data


# In[ ]:





# In[10]:


import csv
import parselmouth
import tgt
import numpy as np
import os

#Work Environment
class PitchMain:


    def Run(self, Wav_file_path, Textgrid_path, tier_name):
        
        fp = FileProcessor()
        pe = PitchExtraction()
        spn = SpeakerNormalization()
        fti = FormatToInterval()
        fo = Formatting()
        t = Training()
        pw = plswrk()

        Wav_file = parselmouth.Sound(Wav_file_path)
            
        data, error, error_arr = fp.iterateTextGridforPitch(Textgrid_path, tier_name, Wav_file)

        file_mean = spn.fileMean(data, "max")
        file_std = spn.fileStd(data, file_mean, "max")
        file_min = spn.fileMin(data, "min")
        file_max = spn.fileMax(data, "max")
        complete_data = spn.zScoreAppend(data, file_mean, file_std, "max")
                
        full_complete_data = pw.contextWindow(complete_data)

        tier_arrays = fti.dictToArr(full_complete_data)

        #print("\n")

        csv_start = os.getcwd()
        csv_base = os.path.basename(Textgrid_path)
        csv_tgt = os.path.join(csv_start, csv_base)
        csv_no_ext = os.path.splitext(csv_tgt)[0]
        csv_file = csv_no_ext + 'Pitch.csv'

        fo.to_csv(tier_arrays, csv_file)
            
        
        return csv_file


# In[ ]:




