import os
json_fname = 'mro_params.json'
json_fname_backup = 'mro_params_bak.json'

# data_path = r"C:\Users\Anand\National University of Ireland, Galway\Group_TOMI Lab - ARANGATH_ANAND\MRO_DATA_NEW_CI_VC\08_12_2020\SNR\1"
# data_path = r"C:\Users\Anand\National University of Ireland, Galway\Group_TOMI Lab - ARANGATH_ANAND\MRO_DATA_NEW_CI_VC\01_01_2020\SNR\1"
data_path = r"C:\Users\anand\National University of Ireland, Galway\Group_TOMI Lab - MRO_DUAL_DET\ATS_DSO\18_03_2021\500_steps"
calib_fname = os.path.join(data_path, '2021.03.18_17.10.34_1.A.bin')

# data_path = r"C:\Users\anand\National University of Ireland, Galway\Group_TOMI Lab - MRO_DATA_NEW_CI_VC (1)\11_12_2020\Imaging\white_tape\3"
# data_path = r"C:\Users\anand\National University of Ireland, Galway\Group_TOMI Lab - MRO_DATA_NEW_CI_VC (1)\09_02_2021\B_frame_mirror"

img_fname = None
# img_fname = os.path.join(data_path,'data153011_000.bin')
# img_fname = os.path.join(data_path,'data194955_000.bin')

# call second backup last to store into image path
json_fname_backup_2 = os.path.join(data_path,"dataBACKUP.json")

acquistion_params = {'Detector gain': 1E4,
                     'digitizer range': '2V'}

manual_params = {'sample_number': 33344,
   'buffer_time': 0.0033,
   'mirror_step_width': 0.003,
   'scan_range': 100.0,
   'lateral_scan_width': 2000.0,  #in micrometers
   'phase_fw': 'phase_fw.npy',
   'phase_rv': 'phase_rv.npy',
   'mirror_cut_off_top': 1400,
   'mirror_cut_off_end': 1100,  #200
      # The img_cut_off seems to be the same for imaging a tilted mirror
   'img_cut_off_top': 550,
   'img_cut_off_end': 1000,
      # The segment length is shorter than 35000 compared to mirror calibration data.
   # 'img_a_line_len': 35000,
   'img_a_line_len': 33344, # was 33325
      # Cut off the first number of samples from the whole B-frame buffer (relates to one or two initial A-lines)
   # 'cut_off_1d': 0,
   'cut_off_1d': 28000,   # was 28000
      # For mirror calibration. In imaging mode this should not have any effect
   'sig_start':1,
   'sig_end': 12,
   'sig_step':2,
   'peak_prominence': 100.0,
     # for the pzt it was 40000000.0. 5000000.0 seems to be good for 20 peaks with VC
   'alpha': 0.5,
   'filter': {'type': 'cheby2',
    'use_num_orders': 12,
    'CF_correction_factor': 2.0,
    'gpass': 0.001,
    'gstop': 60,
    'half_passbandwidth_px': 55,
    'half_stopbandwidth_px': 75,
    'use_initial_sos_cond': False},
   'resample_to': 5000,
   'resample_to_vertically': 1000,
   'resample_to_before_hilbert': 5000,
   'fft_fw_sum': 'fft_fw_sum.npy',
   'fft_fw_sum_svg': 'fft_fw_sum_svg.npy',
   'fft_rv_sum': 'fft_rv_sum.npy',
   'fft_rv_sum_svg': 'fft_rv_sum_svg.npy',
     # Initial set these values to zero
     # This controls the shift of segments to optimze A-line assembling
   'steps_correlate_orders': 3,
   'aline_pad_correction_fw': 0,
   'aline_pad_correction_rv': 0,
     # Use cutting 'cut'.
     # Other method could be 'summing', 'mean', but those may produce unexpected results. Idea is to use SNR improvement of the overlap.
   'aline_merge_method': 'cut',
     # If 'cut' then this allows to define the cut position
   'cut_forward': 100,
   'subtract_median_background_mirror': False,
   'subtract_median_background_image': False,
   'nan_value': 1e-10,
   'fw_resample': 1.0,
   'fw_roll': -1,
   'use_fw_for_rv_phase': False}

import json
params = json.load(open('mro_params.json','r'))
params['manual_params'] = manual_params
json.dump(params, open('mro_params.json', 'w'), indent=2)