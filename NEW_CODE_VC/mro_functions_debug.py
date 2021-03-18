"""
This provides wrapper functions to include all kinds of analysis plots.

Usage:
    Import this module like
from anand_mro_processing import mro_functions_debug as mf

    instead of (or commenting it out)
from anand_mro_processing import mro_functions as mf
"""
from NEW_CODE_VC import mro_plotting as mp
from NEW_CODE_VC import mro_functions as mf
from NEW_CODE_VC.mro_functions import json
from NEW_CODE_VC.mro_functions import np

# We use currentframe to extract stack data!
# According to https://docs.python.org/3/library/inspect.html
# this may not work on all platforms.
import inspect

# Value n means execute the function normally.
# Value 0 means the function not executed.
# Value s means the function is storing temp data.
# Value t means the function is not executed but loads temp data instead.
#

call_func  = {'load_data':                       'n',
              'align_mirror_data_and_save_phase':'n',
              'align_img_data_fw_rv':            'n',
              'linearize_signal':                'n',
              'update_json_filter_frequencies':  'n',
              'signal_apodization':              'n',
              'resample_signal':                 'n',
              'filter_signal':                   'n',
              'hilbert_all_aline_orders':        'n',
              'assemble_orders_to_alines':       'n'
              }

# Other more compact way to set bool values. Overwrite previous settings.
call_pattern   = 'nnnnnnnnnn'
# call_pattern = 'nnnnnnnnnn'
# call_pattern = 'nnnnnnnsnn'
# call_pattern = '0000000tnn'
# call_pattern = '0000000tsn'
# call_pattern = '00000000tn'
# call_pattern = 'nnnnnnnnsn'
# call_pattern = '00000000tn'
call_func = {k:v for k,v in zip(call_func.keys(),list(call_pattern))}

def load_json_params( time_stamp ):
    params = mf.load_json_params( time_stamp )
    params['auto_params']['debug_mode'] = True
    json.dump(params, open(params['auto_params']['json_fname'], 'w'), indent=2)
    return params

def backup_params( params, json_fname_backup, time_stamp):
    mf.backup_params(params, json_fname_backup, time_stamp)


def load_data(params, calib_fname, img_fname):
    if call_func['load_data'] == 't':
        data = np.load('data.npy')
        return params, data
    if call_func['load_data'] == '0':
        return params, None
    if call_func['load_data'] == 'n':
        params, data = mf.load_data(params, calib_fname, img_fname)
        # mp.plt.figure(num='1D data')
        # mp.plt.plot( data[0:100:10].T)
        # mp.plt.pause(0.1)
        # mp.plt.show()
        return params, data
    if call_func['load_data'] == 's':
        params, data = mf.load_data(params, calib_fname, img_fname)
        np.save('data.npy',data)
        return params, data


def align_mirror_data_and_save_phase(params, data):
    if call_func['align_mirror_data_and_save_phase'] == 't':
        raise NotImplementedError
    if call_func['align_mirror_data_and_save_phase'] == 's':
        raise NotImplementedError
    if call_func['align_mirror_data_and_save_phase'] == '0':
        return params
    if call_func['align_mirror_data_and_save_phase'] == 'n':
        debug_dict = {}#{'do_plot':True, 'do_stop':True}
        debug_dict = {'do_plot':True, 'do_stop':True}
        params = mf.align_mirror_data_and_save_phase(params, data, debug_dict)
        debug_dict = {}#{'do_plot':True, 'do_stop':True}
        debug_dict = {'do_plot':True, 'do_stop':True}
        mp.plot_non_linear_phase(params, debug_dict)
        return params


def align_img_data_fw_rv(params, data):
    if call_func['align_img_data_fw_rv'] == '0':
        return params, None, None
    if call_func['align_img_data_fw_rv'] == 'n':
        debug_dict = {'do_plot':False}
        params, data_aligned_fw, data_aligned_rv = mf.align_img_data_fw_rv(params, data, debug_dict)
        # plot
        return params, data_aligned_fw, data_aligned_rv


def linearize_signal(params=None, data_aligned=None, phase_type=None):
    if call_func['linearize_signal'] == '0':
        return params, None
    if call_func['linearize_signal'] == 'n':
        return_data = mf.linearize_signal(params=params, data_aligned=data_aligned, phase_type=phase_type)
        if hasattr(linearize_signal,'call_count'):
            linearize_signal.call_count += 1
        else:
            linearize_signal.call_count = 1
        # Try to access stack data
        # print(inspect.getframeinfo(inspect.currentframe()))
        # print(inspect.getframeinfo(inspect.currentframe().f_back))
        frame = inspect.getmembers(inspect.currentframe().f_back)
        # print(frame)
        # This is a hack. Not sure if -4 and -1 is always sure
        linear_fw_data = frame[-4][-1].get('linear_fw_data')
        if linearize_signal.call_count > 1:
            assert linear_fw_data is not None, 'No linear_fw_data found'

        if linear_fw_data is not None:
            linear_rv_data = return_data[1]

            debug_dict = {'do_plot':False}
            mp.plot_linear_signal(params, linear_fw_data, linear_rv_data, debug_dict)
            # if params['auto_params']['is_calibration']:
            debug_dict = {'do_plot':True}
            mp.plot_linear_phase(params, linear_fw_data, linear_rv_data, debug_dict)
        return return_data


def update_json_filter_frequencies(params, linear_fw_data, linear_rv_data):
    if call_func['update_json_filter_frequencies'] == '0':
        return params
    if call_func['update_json_filter_frequencies'] == 'n':
        params = mf.update_json_filter_frequencies(params, linear_fw_data, linear_rv_data)
        debug_dict = {'do_plot': True, 'do_wait': True}
        mp.plot_fft_sum_and_peaks(params, debug_dict)
        debug_dict = {'do_plot': True, 'start_at': 0}
        mp.plot_linear_signal_and_do_fft(params,linear_fw_data,linear_rv_data, debug_dict)
        return params


def signal_apodization(params, linear_data, process_type=None):
    if call_func['signal_apodization'] == '0':
        return params, None
    if call_func['signal_apodization'] == 'n':
        return mf.signal_apodization(params=params, linear_data=linear_data, process_type=process_type)


def resample_signal(params, linear_data, process_type):
    if call_func['resample_signal'] == 't':
        linear_data_rs = np.load('lin_data_rs_'+process_type+'.npy')
        return params, linear_data_rs
    if call_func['resample_signal'] == '0':
        return params, None
    if call_func['resample_signal'] == 'n':
        return mf.resample_signal(params=params, signal=linear_data, process_type=process_type)
    if call_func['resample_signal'] == 's':
        params, linear_data_rs = mf.resample_signal(params=params, signal=linear_data, process_type=process_type)
        np.save('lin_data_rs_'+process_type,linear_data_rs)
        return params, linear_data_rs

def filter_signal(params, linear_data, process_type):
    debug_dict = {
        'do_plot_filtered_sig': False,
        'do_plot_filtered_sig_wait': False,
        'do_plot_filter_response': False,
        'do_plot_filter_and_sig': False,
        'do_plot_filter_and_sig_wait': True,
        'show_each_filter': False,
        'wait_each_filter': False,
        'start_at': 28}
    if call_func['filter_signal'] == 't':
        aline_orders = np.load('aline_orders_'+process_type+'.npy')
        return params, aline_orders
    if call_func['filter_signal'] == '0':
        return params, None
    if call_func['filter_signal'] == 'n':
        return mf.filter_signal(params, linear_data, process_type, debug_dict)
    if call_func['filter_signal'] == 's':
        params, aline_orders = mf.filter_signal(params, linear_data, process_type, debug_dict)
        np.save('aline_orders_'+process_type, aline_orders)
        return params, aline_orders


def hilbert_all_aline_orders(params, aline_orders, process_type):
    if call_func['hilbert_all_aline_orders'] == 't':
        aline_orders_hlb = mf.np.load('aline_orders_hlb_'+process_type+'.npy')
        return_data = params, aline_orders_hlb
    if call_func['hilbert_all_aline_orders'] == '0':
        return params, None
    if call_func['hilbert_all_aline_orders'] == 'n':
        return_data = mf.hilbert_all_aline_orders(params, aline_orders, process_type)
    if call_func['hilbert_all_aline_orders'] == 's':
        # save temp data
        return_data = mf.hilbert_all_aline_orders(params, aline_orders, process_type)
        aline_orders_hlb = return_data[1]
        mf.np.save('aline_orders_hlb_'+process_type, aline_orders_hlb)

    if params['auto_params']['is_calibration']:
        # Those plots make only sense for mirror calibration data.
        # The correlators will fail if there are insufficient data points.
        # mp.plot_raw_aline_orders_hilbert(params, aline_fw_orders_hlb)
        # mp.plot_orders_scaled(params,aline_fw_orders_hlb)
        # mp.plot_summing_of_correlation_values(params, aline_fw_orders_hlb)
        if 'fw' in process_type:
            is_reversed = False
        elif 'rv' in process_type:
            is_reversed = True

        debug_dict = {'show_correlation':True, 'show_all_orders': False, 'show_brief': False, 'do_stop':False, 'do_stop2':False}
        mp.plot_orders_spatially_aligned(params,
                                         aline_orders_hlb= return_data[1],
                                         is_reversed=is_reversed,
                                         process_type = process_type,
                                         debug_dict = debug_dict)

    return return_data


def assemble_orders_to_alines(params, aline_orders_hlb, is_reversed, process_type):
    if call_func['assemble_orders_to_alines'] == 't':
        bframe = mf.np.load('bframe_'+process_type+'.npy')
        return params, bframe
    if call_func['assemble_orders_to_alines'] == '0':
        return params, None
    if call_func['assemble_orders_to_alines'] == 'n':
        return mf.assemble_orders_to_alines(params, abs(aline_orders_hlb), is_reversed, process_type,{'plot_each_order':False})
    if call_func['assemble_orders_to_alines'] == 's':
        return_data = mf.assemble_orders_to_alines(params, abs(aline_orders_hlb), is_reversed, process_type)
        mf.np.save('bframe_'+process_type,return_data[1])
        return return_data

