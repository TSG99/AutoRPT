#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import argparse
import Intensity_ex
from Intensity_ex import IntensityMain
import Pitch_ex
from Pitch_ex import PitchMain
import CompleteIntensityModel
from CompleteIntensityModel import IBM
import CompletePitchModel
from CompletePitchModel import PBM, PPM
import TextgridCreate
from TextgridCreate import TextgridCreation

def main(Textgrid_path, Wav_file, tier_name):
    pm = PitchMain()
    im = IntensityMain()
    ibm = IBM()
    pbm = PBM()
    ppm = PPM()
    tgc = TextgridCreation()

    pm_csv_file = pm.Run(Wav_file, Textgrid_path, tier_name)
    im_csv_file = im.Run(Wav_file, Textgrid_path, tier_name)
    csv_file_ibm = ibm.ibm_model(im_csv_file)
    csv_file_pbm = pbm.pbm_model(pm_csv_file)
    csv_file_ppm = ppm.ppm_model(pm_csv_file)
    tgc.outputTGT(Textgrid_path, csv_file_ibm, 'Bound-Pred', 'IBM')
    tgc.outputTGT(Textgrid_path, csv_file_ppm, 'Prom-Pred', 'PPM')
    tgc.outputTGT(Textgrid_path, csv_file_pbm, 'Bound-Pred', 'PBM')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Textgrid and Wav files with various models.')
    parser.add_argument('--textgrid', type=str, required=True, help='Path to Textgrid file')
    parser.add_argument('--wav', type=str, required=True, help='Path to Wav file')
    parser.add_argument('--tier', type=str, required=True, help='Name of the target tier')

    args = parser.parse_args()

    main(args.textgrid, args.wav, args.tier)



# %%
