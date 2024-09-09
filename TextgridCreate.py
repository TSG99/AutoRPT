#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import tgt

class TextgridCreation:
    #Function to read intervals and labels from CSV
    def read_intervals_from_csv(self, csv_file_path, usecols):
        df = pd.read_csv(csv_file_path, usecols=usecols)
        intervals = df.values.tolist()  
        return intervals

    def outputTGT(self, input_textgrid_path, csv_file_path, feat, bop):
        #Load the existing TextGrid file
        tg = tgt.io.read_textgrid(input_textgrid_path)

        #Define the columns to read from the CSV file
        csv_start = os.getcwd()
        usecols = ['Start', 'End', 'predictions'] 

        #Read intervals and labels from CSV
        intervals = self.read_intervals_from_csv(csv_file_path, usecols)

        #Create a new interval tier
        new_tier_name = feat
        new_tier = tgt.IntervalTier(name=new_tier_name)

        #Sort intervals by start time to ensure proper order
        intervals.sort(key=lambda x: x[0])

        #Add intervals to the new tier without overlapping
        for start_time, end_time, label in intervals:
            start_time = float(start_time)
            end_time = float(end_time)
            if new_tier.intervals:
                last_interval = new_tier.intervals[-1]
                if start_time < last_interval.end_time:
                    start_time = last_interval.end_time
            if start_time < end_time:
                new_interval = tgt.Interval(start_time, end_time, str(label))
                new_tier.add_interval(new_interval)

        #Add the new tier to the TextGrid
        tg.add_tier(new_tier)

        output_name = os.path.basename(input_textgrid_path)
        output_stem = os.getcwd()
        output = output_stem + "\\tg_outputs\Predictions_" + bop + "_" + output_name

        tgt.io.write_to_file(tg, output, format='long')

       # print(f"New interval tier '{new_tier_name}' added and saved to '{output}'")

