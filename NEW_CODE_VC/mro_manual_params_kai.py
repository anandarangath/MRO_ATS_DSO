import os
json_fname = 'mro_params.json'
json_fname_backup = 'mro_params_bak.json'

data_path = '/Users/kai/National University of Ireland, Galway/Group_TOMI Lab - 08_12_2020/SNR/1/'
calib_fname = os.path.join(data_path, 'data.bin')

# data_path = '/Users/kai/National University of Ireland, Galway/Group_TOMI Lab - 08_12_2020/SNR/B_frame/1/'
# img_fname = os.path.join(data_path,'B-frame_mirror_t1134817_000.bin')
# data_path = '/Users/kai/National University of Ireland, Galway/Group_TOMI Lab - 08_12_2020/SNR/B_frame/alines_100'
# img_fname = os.path.join(data_path,'B-frame_mirror_t1170343_000.bin')
# img_fname = os.path.join(data_path,'B-frame_mirror_z12_5700170916_000.bin')
data_path = '/Users/kai/National University of Ireland, Galway/Group_TOMI Lab - 08_12_2020/SNR/B_frame/tilt_mirror'
img_fname = os.path.join(data_path,'B-frame_tilt_M_1200528_000.bin')
# img_fname = None
json_fname_backup_2 = os.path.join(data_path,"params_bak_kai.json")


manual_params = {'sample_number': 35000,
   'buffer_time': 0.0033,
   'mirror_step_width': 0.003,
   'axial_scan_depth': 1.0,
   'lateral_scan_width': 1.0,
   'phase_fw': 'phase_fw.npy',
   'phase_rv': 'phase_rv.npy',
   'mirror_cut_off_top': 550,
   'mirror_cut_off_end': 1000,
   'img_cut_off_top': 550,
   'img_cut_off_end': 1000,
      # This new parameter accounts for a different A-line length compared to sample_number.
      # Ideally it would be the same as sample_number but most likely is not.
   'img_a_line_len': 33325,
   'cut_off_1d': 28000,
   'sig_start':1,
   'sig_end': 16,
   'sig_step':2,
   'peak_prominence': 5000000.0,
     # for the pzt it was 40000000.0. 5000000.0 seems to be good for 20 peaks with VC
   'alpha': 0.5,
   'filter': {'type': 'ellip',
    'CF_correction_factor': 2.0,
    'gpass': 0.001,
    'gstop': 60,
    'half_passbandwidth_px': 35,
    'half_stopbandwidth_px': 55,
    'use_initial_sos_cond': False},
   'resample_to': 5000,
   'resample_to_vertically': 1000,
   'resample_to_before_hilbert': 5000,
   'fft_fw_sum': 'fft_fw_sum.npy',
   'fft_fw_sum_svg': 'fft_fw_sum_svg.npy',
   'fft_rv_sum': 'fft_rv_sum.npy',
   'fft_rv_sum_svg': 'fft_rv_sum_svg.npy',
   'steps_correlate_orders': 3,
     # Initial set these values to zero
     # This controls the shift of segments to optimze A-line assembling
   'aline_pad_correction_fw': -180,
   'aline_pad_correction_rv': -180,
     # Use cutting 'cut'.
     # Other method could be 'summing', 'mean', but those may produce unexpected results. Idea is to use SNR improvement of the overlap.
   'aline_merge_method': 'cut',
     # If 'cut' then this allows to define the cut position
   'cut_forward': 60,
   'subtract_median_background_mirror': False,
   'subtract_median_background_image': False,
   'nan_value': 1e-10,
   'fw_resample': 1.0,
   'fw_roll': -1,
   'use_fw_for_rv_phase': False}

import json
params = json.load(open('mro_params.json','r'))
params['manual_params'] = manual_params
json.dump(params, open('mro_params.json','w'), indent=2)