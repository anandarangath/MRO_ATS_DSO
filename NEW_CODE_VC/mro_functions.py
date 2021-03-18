import numpy as np

import scipy.signal as sg
import scipy.fftpack as spfft
import scipy.interpolate as ip
import json
from json import JSONDecoder
import warnings
from warnings import warn

from NEW_CODE_VC import mro_plotting as mp
from NEW_CODE_VC.mro_manual_params import *
import os

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

def load_json_params( time_stamp ):
    params = json.load(open(json_fname))
    backup_params(params, time_stamp)
    params['auto_params']['json_fname'] = json_fname
    params['auto_params']['time_stamp'] = time_stamp
    if img_fname is not None:
        params['auto_params']['title_str'] = mp.guess_str_from_path(img_fname)
    else:
        params['auto_params']['title_str'] = mp.guess_str_from_path(calib_fname)

    params['auto_params']['debug_mode'] = False
    # update debug_mode status
    json.dump(params, open(json_fname, 'w'), indent=2)
    return params


def backup_params( params, time_stamp):
    for fname in (json_fname_backup, json_fname_backup_2):
        if not os.path.exists(fname):
            json.dump({time_stamp: params}, open(fname,'w'),indent=2)
        else:
            params_bak = json.load(open(fname))
            params_bak[time_stamp] = params
            json.dump(params_bak , open(fname,'w'), indent=2)


def load_data(params, calib_fname, img_fname):
    """
    Load data
    :param params:
    :return:
    """

    # if img_fname is given then we assume processing of an image.
    # otherwise we assume calibration with mirror data.
    if img_fname is not None:
        print('Perform image processing.')
        is_calibration = False
        fname = img_fname
        sample_number = params['manual_params']['img_a_line_len']
        cut_off_1d = params['manual_params']['cut_off_1d']
        with open(fname,'rb') as fid:
            bytewidth = 2
            fid.seek(cut_off_1d * bytewidth, os.SEEK_SET)
            data = np.fromfile(fid,('>u2', sample_number))
    else:
        print('Perform calibration.')
        is_calibration = True
        fname = calib_fname

        sample_number = params['manual_params']['sample_number']
        # with open(fname,'rb') as fid:
        #     data = np.fromfile(fid,('>u2', sample_number))
        data= np.fromfile(open(fname, 'rb'), dtype=('f8', sample_number))

    data = data - np.mean(data)
    print("The shape of the data is ", np.shape(data))

    params['auto_params'].update( {
        'is_calibration': is_calibration,
        'calib_fname':calib_fname,
        'img_fname':img_fname
        })
    # Lets save here as we may change the path above
    json.dump(params,open(params['auto_params']['json_fname'],'w'),indent=2)
    return params, data


def align_mirror_data_and_save_phase(params, data, debug_dict={}):
    """
    Use this function to visualize the phase fitting to the mirror data.
    The cut_off positions are currently used for forward and reverse equally,
    but we save the phase fit for each (see code line below):
        np.save('phase_fw.npy',phase_fw)
        np.save('phase_rv.npy',phase_rv)

    :param data:
    :param params:
    :return:
    """
    # If image processing we can not do phase fitting
    if params['auto_params']['is_calibration'] is False:
        print('Skip fit phase; Image processing.')
        return params # leave this function right here

    sample_number_given = params['manual_params']['sample_number']
    sample_number_data = data.shape[1]
    assert sample_number_data == sample_number_given, 'Mismatch of samples should not happen here!'
    sample_number = sample_number_data

    man_ps = params['manual_params']
    sig_start = man_ps['sig_start']
    sig_end = man_ps['sig_end']
    sig_step = man_ps['sig_step']

    mirror_cut_off_top = man_ps['mirror_cut_off_top']
    mirror_cut_off_end = man_ps['mirror_cut_off_end']

    data_fw = data[sig_start:sig_end:sig_step, mirror_cut_off_top:sample_number // 2 - mirror_cut_off_end]
    data_rv = data[sig_start:sig_end:sig_step, mirror_cut_off_top+sample_number // 2 :- mirror_cut_off_end]

    data_fw_sum = sum(data_fw)
    data_rv_sum = sum(data_rv)
    phase_fw = np.unwrap(np.angle(sg.hilbert(data_fw_sum)))
    phase_fw = sg.savgol_filter(x=phase_fw[::1], window_length=501, polyorder=1)
    phase_rv = np.unwrap(np.angle(sg.hilbert(data_rv_sum)))
    phase_rv = sg.savgol_filter(x=phase_rv[::1], window_length=501, polyorder=1)
    if params['manual_params']['use_fw_for_rv_phase']:
        phase_rv = phase_fw
    if debug_dict.get('do_plot'):
        mp.plot_mirror_data_fw_rv(params, data_fw, data_rv, debug_dict)

    np.save(man_ps['phase_fw'],phase_fw)
    np.save(man_ps['phase_rv'],phase_rv)

    return params

def align_img_data_fw_rv(params, data, debug_dict={}):
    """
        Important: Mirror data and image data must be aligned both!!!
        Meaning image data must also be aligned because it has different time points and sample lengths.

    To plot the orders for alignment set the parameter 'img_calibration = True'.

    For image data a glass slide with a sufficient angle should be used to have sufficient visible signal for alignment.

    The image alignment may need to be repeated as soon as acquisition parameters are changed.
    :param mps:
    :param data: The data from the binary file.
    :param debug_dict: {'do_plot':BOOL, 'do_wait':BOOL}
    :return:
    """

    # data should contain the image data if is_calibration = False
    # sample_number = params['sample_number']
    # TODO: Draw sequence flow-chart and document: image alignment needs to cut-off the mirror data again.
    # align_mirror_data does cut-off only for the phase data.

    data_samples = data.shape[1] # samples per data segment
    # phase_fw = np.load(params['phase_fw'])
    # phase_rv = np.load(params['phase_rv'])
    if params['auto_params']['is_calibration']:
        sample_number = data.shape[1]
        man_pars = params['manual_params']
        mirror_cut_off_top = man_pars['mirror_cut_off_top']
        mirror_cut_off_end = man_pars['mirror_cut_off_end']

        data_aligned_fw = data[:, mirror_cut_off_top : sample_number // 2 - mirror_cut_off_end]
        data_aligned_rv = data[:, mirror_cut_off_top + sample_number // 2 : sample_number - mirror_cut_off_end]
        return params, data_aligned_fw, data_aligned_rv

    else:
        # TODO: image alignment: are our assumptions still true?
        # We receive a B-frame as a continuous array due to the problem that it can have a different segment length.
        # For this purpose we deal with it as a 1D array ???
        # The mirror data are available as 2D array and in this case they should be aligned using their mirror alignment.
        # For B-frame processing the data have to be converted to 2D anyway which can be used for alignment computation.
        # That means as soon as we get a B-frame 1D array we convert it to 2D using reshape and
        # the reshape action is used to align the image data for optimal linearization.
        man_pars = params['manual_params']

        img_cut_off_top = man_pars['img_cut_off_top']
        img_cut_off_end = man_pars['img_cut_off_end']
        img_a_line_len = man_pars['img_a_line_len']

        # TODO: this division will not be perfect and depending on data1D_len some data will be in an odd-length array.

        data_aligned_fw = data[:, img_cut_off_top : img_a_line_len // 2 - img_cut_off_end]
        data_aligned_rv = data[:, img_cut_off_top + img_a_line_len // 2 : img_a_line_len - img_cut_off_end]

        # mp.plot_align_img_data_debug(params, data)
        if debug_dict.get('do_plot'):
            # mp.plt.figure(num='Align img orders')
            fig,ax = mp.subplots(1,2)
            ax[1].set_title('Align img ordres\n Close to proceed!')
            ax[0].plot( data_aligned_fw[50:53].T)
            ax[1].plot( data_aligned_rv[50:53].T)
            mp.plt.show()

    return params, data_aligned_fw, data_aligned_rv

def linearize_signal(params, data_aligned, phase_type):
    """
    Linearize aligned data with phase.

    See also
    --------
    function `align_img_data_fw_rv`

    Params
    ------

    :param params:
    :param data_aligned: You get aligned data from function `align_img_data_fw_rv`.
    :param phase:
    :return: params, lin_data

    """
    data_order_num = data_aligned.shape[0]
    data_len = data_aligned.shape[1]
    phase = mp.np.load('phase_'+phase_type+'.npy')
    phase_len = phase.shape[0]

    if data_len != phase_len:
        warn('Resampling nonlin phase to image samples from {} to {} samples.'.format(phase_len, data_len))
    phase = sg.resample(x=phase,num=data_len) # match samples to image/phase
    # sg.savgol_filter()
    phase = phase/max(phase)*data_len # match sample range to image

    interpolation_fw = ip.interp1d(phase,data_aligned) # make function
    lin_samples = range(np.ceil(phase.min()).astype(np.int),data_len) # make linear sample range
    linear_data = interpolation_fw(lin_samples) # find / interpolate samples for linear samples

    return params, linear_data


def update_json_filter_frequencies(params, linear_fw_data, linear_rv_data):
    """
    Find filter center frequencies with peak detection and store those into the json parameter file.
    For forward and reverse each a frequency list is generated.
    To equalize the number of found orders (frequencies) the missing ones will be completed from the other.
    :param params:
    :param linear_fw_data:
    :param linear_rv_data:
    :return:
    """
    # If image processing we can not do peak detection
    if params['auto_params']['is_calibration'] is False:
        print('Skip peak detection ... we do image processing with stored peak values.')
        return params # leave this function right here

    fft_fw_sig = spfft.fftshift(spfft.fft(linear_fw_data,axis=1))
    fft_fw_sum = np.sum(abs(fft_fw_sig), axis=0)[fft_fw_sig.shape[1]//2:]
    fft_fw_sum_svg = sg.savgol_filter(x=fft_fw_sum, window_length=15, polyorder=1)  # Experimental

    fft_rv_sig = spfft.fftshift(spfft.fft(linear_rv_data,axis=1))
    fft_rv_sum = np.sum(abs(fft_rv_sig), axis=0)[fft_rv_sig.shape[1]//2:]
    fft_rv_sum_svg = sg.savgol_filter(x=fft_rv_sum, window_length=15, polyorder=1)  #  Experimental

    # debug_dict={'start_at':400}
    # mp.plot_debug_json_freq_update(linear_fw_data, linear_rv_data, debug_dict)

    peaks_fw, _ = sg.find_peaks(fft_fw_sum_svg, prominence=params['manual_params']['peak_prominence'])
    peaks_rv, _ = sg.find_peaks(fft_rv_sum_svg, prominence=params['manual_params']['peak_prominence'])

    ### Analysis of the goodness of frequency detection and generation of warnings if it is not good.
    # clear previous warnings
    ap = params['auto_params']
    if ap.get('warn_peaks_array'): ap.pop('warn_peaks_array')
    if ap.get('warn_peaks_freq_diff'): ap.pop('warn_peaks_freq_diff')
    if abs(len(peaks_fw) - len(peaks_rv)) > 3:
        # add warning
        ap['warn_peaks_array']='Large length difference between fw and rv ({}, {})!'.format(len(peaks_fw),len(peaks_rv))
        warn('Peak array length large difference!')

    if abs(np.diff(peaks_fw).mean() - np.diff(peaks_rv).mean()) > 1:
        # add warning
        ap['warn_peaks_freq_diff']='Large frequency difference between fw and rv ({:.1f}, {:.1f})!'.format(np.diff(peaks_fw).mean(),np.diff(peaks_rv).mean())
        warn('Peak frequency difference large!')

    # synchronize length of peak array
    # ignoring elements that are in excess of num_peaks
    num_peaks = np.min((len(peaks_rv), len(peaks_fw)))
    peaks_fw = peaks_fw[:num_peaks]
    peaks_rv = peaks_rv[:num_peaks]

    # fill missing frequencies taking from the longest list
    # If the fw list as more peaks take the peak from there and use it rv (or vice versa)
    # if len(peaks_fw) > len(peaks_rv):
    #     missing_idx = range(len(peaks_rv), len(peaks_rv) + (len(peaks_fw) - len(peaks_rv)))
    #     peaks_rv = np.concatenate((peaks_rv, peaks_fw[missing_idx] ))
    # elif len(peaks_fw) > len(peaks_rv):
    #     missing_idx = range(len(peaks_fw), len(peaks_fw) + (len(peaks_rv) - len(peaks_fw)))
    #     peaks_fw = np.concatenate((peaks_fw, peaks_rv[missing_idx]))

    params['auto_params'].update({
        'peaks_fw_px': str(peaks_fw.tolist()),
        'peaks_rv_px': str(peaks_rv.tolist()),
    })
    json.dump(params,open(params['auto_params']['json_fname'],'w'),indent=2)

    mpms = params['manual_params']
    # np.save(mpms['fft_fw_sum'],fft_fw_sum)
    np.save(mpms['fft_fw_sum_svg'], fft_fw_sum_svg)
    # np.save(mpms['fft_rv_sum'],fft_rv_sum)
    np.save(mpms['fft_rv_sum_svg'], fft_rv_sum_svg)

    # Use the saved fft_fw/rv_sum_svg and plot them including the peaks detected
    if not ap['debug_mode'] and (ap.get('warn_peaks_array') or ap.get('warn_peaks_freq_diff')):
        mp.plot_fft_sum_and_peaks(params, {'do_plot': True, 'do_wait': True})

    return params


def signal_apodization(params, linear_data, process_type=None):

    alpha = params['manual_params']['alpha']
    segment_len  = linear_data.shape[1]

    # mp.plot_apodization(linear_data)

    linear_data = linear_data[:] * sg.windows.tukey(M=segment_len, alpha=alpha)

    return params, linear_data


def resample_signal(params, signal, process_type = None):
    if signal.ndim < 3:
        axis = 1
        resample_to = params['manual_params']['resample_to']
    elif signal.ndim == 3:
        axis =2
        resample_to = params['manual_params']['resample_to_before_hilbert']

    before_smpls = signal.shape[axis]

    params['auto_params']['segment_len_before_resample_'+process_type] = before_smpls
    json.dump(params, open(params['auto_params']['json_fname'],'w'), indent=2)

    print('Resample from {} to {}'.format(before_smpls, resample_to))

    signal = sg.resample(x=signal, num=resample_to, axis=axis)

    print('signal shape ',signal.shape)
    return params, signal


def create_new_BW_data_to_json(params):
    """
    This function generates some json parameters.
    The list of parameters must be manually checked here in the code
    before calling.
    To activate the function the return statement must be commented.
    :param params:
    :return:
    """
    # Call this only to create new values into the json file.
    return # safety not to call by accident
    params = params['manual_params']
    params['filter'] = {'type':'ellip'}
    params['filter']['CF_correction_factor'] = 2.15
    params["filter"]["gstop"] = 90 # 80 to 90 dB depending on the noise power. Larger values slow down processing.
    params["filter"]["gpass"] = 0.1 # Pass-band ripple 0.1 dB. Needs to be confirmed.
    params['filter']['half_passbandwidth_px'] = 60
    params['filter']['half_stopbandwidth_px'] = 50
    json.dump(params,open(params['auto_params']['json_fname'],'w'),indent=2)
# create_new_BW_data_to_json()


def make_filter_response_array(peaks, filters, sig_len, debug_dict):
    # Compute filter response for each sos-filter (from filters)
    filter_response = []
    len_aline = sig_len
    worN = len_aline
    for flt in filters:
        sosdata = flt['sosdata']
        # h = 1.
        # for row in sosdata:
        #     w, rowh = sg.freqz(row[:3], row[3:], worN=worN, whole=True)
        #     h *= rowh
        w, h = sg.sosfreqz(sosdata, worN=len_aline, whole=True)
        filter_response.append({'CP_fw': flt['CP_fw'], 'H': h})

        mp.plot_filter_response_each(peaks, h, debug_dict)

    return filter_response


def filter_signal(params, linear_data, process_type=None, debug_dict={}):
    """
    Filter linear_data and return new array with separate orders.

    I.e. if input data look like:
    linear_data = [num_alines, num_samples]

    output array looks like:
    aline_orders = [num_alines, num_orders, num_samples]

    Please note that we use here internal plotting functions that can be enabled if required for analysis.
    To enable plotting find the function in the code and set do_plot = True
    mp.plot_filter_response_each(h,do_plot=False,do_wait=True)
    mp.plot_filter_response_vs_signal_in(filter_response, linear_fw_data, do_plot=False, do_wait=True)
    mp.plot_signal_filtered_each(aline_fw, fsigs, filter_response, do_plot=False  , do_wait=True)

    :param params:
    :param linear_data:
    :param process_type: string 'fw' or 'rv' ... forward and reverse respectively.
    :return:
    """
    assert type(process_type) is str, 'Please set process_typ to \'fw\' or \'rv\'!'
    print('filtering ({}):'.format(process_type))

    peaks_str = params['auto_params']['peaks_' + process_type + '_px'].strip('[]')
    correction_factor = params['manual_params']['filter']['CF_correction_factor']
    peaks = np.fromstring( peaks_str, sep=',' ) * correction_factor
    resample_factor = 1 #params['resample_to']/params['segment_len_before_resample_'+process_type]
    peaks = np.round(peaks * resample_factor).astype(np.int)
    # as a site note eval would have worked as well but
    # reading the string explicitly is safer and freedom of mind.
    # “Advanced Iterators - Dive Into Python 3.” https://diveintopython3.net/advanced-iterators.html#eval (accessed Jul. 22, 2020).

    half_passbandwidth_px = params['manual_params']['filter']['half_passbandwidth_px'] * resample_factor# 60 sample units
    half_stopbandwidth_px = params['manual_params']['filter']['half_stopbandwidth_px'] * resample_factor # 50 sample units
    HPW = half_passbandwidth_px * resample_factor
    HSW = half_stopbandwidth_px * resample_factor
    bandwidths = []
    # iirdesign assumes that the highest frequency based on the sample number is 1.0 and
    # the lowest frequency is 0.0.
    # We call those filter frequency units
    # Create pass-band (WP) and stop-band (WS) values in sample units
    # Note that we multiply by TWO to ...
    segment_len = linear_data.shape[1]
    HPW /= segment_len
    HSW /= segment_len
    use_num_orders = params['manual_params']['filter']['use_num_orders']
    if len(peaks)<use_num_orders:
        warn('Parameter use_num_orders = {} is larger than peak len {}'.format(use_num_orders, len(peaks)))
        use_num_orders = len(peaks)-1
    for peak in peaks[0:use_num_orders]:
        peak /= segment_len
        peak *= 1.00
        # TODO Peak correction seems not improve things at the moment. So peak *= 1.00 may be removed sometimes.
        freq_dep_factor = 1 # frequency dependent factor
        bandwidths.append({'WP':np.array([peak-HPW,     peak+HPW])    /freq_dep_factor,
                           'WS':np.array([peak-HPW-HSW, peak+HPW+HSW])/freq_dep_factor})

    # [print(bw) for bw in bandwidths]
    # [print(np.diff(bw['WS'])) for bw in bandwidths]


    # Design filters for each peak
    filters = []
    for BW,CP in zip(bandwidths,peaks):
        WP = BW['WP']
        WS = BW['WS']
        # print(WP, WS)
        sosdata = sg.iirdesign(wp=WP,
                               ws=WS,
                               gpass=params['manual_params']["filter"]['gpass'],
                               gstop=params['manual_params']["filter"]['gstop'],
                               ftype=params['manual_params']["filter"]['type'],
                               output='sos',
                               analog=False)
        # Compute initial conditions
        zi = sg.sosfilt_zi(sosdata)
        filters.append({'CP_fw':CP, 'sosdata':sosdata, 'zi':zi})

    if debug_dict.get('do_plot_filter_response'):
        mp.plot_filter_response_vs_signal_in(peaks, filters, linear_data, debug_dict)

    aline_orders = []
    print('{: 4d}'.format(0),end='')

    if debug_dict.get('do_plot_filtered_sig'):
        start_at = debug_dict.get('start_at')
        if linear_data.shape[0] < start_at:
            start_at = linear_data.shape[0]//4
    else:
        start_at = 0

    # Apply filter to all Segments
    for n,segment in enumerate(linear_data[start_at:]):
        fsigs = [] # filtered signals according to number of peaks_fw
        for flt in filters:
            sosdata = flt['sosdata']
            zi = flt['zi']
            if params['manual_params']['filter']['use_initial_sos_cond']:
                fsig, _ = sg.sosfilt(sosdata, segment, zi=zi)
            else:
                fsig = sg.sosfiltfilt(sosdata, segment)
            fsigs.append(fsig)

        # Collect filtered signals
        aline_orders.append(fsigs)

        if np.mod(n,10) == 0: print('\b\b\b\b{: 4d}'.format(n),end='')

        if debug_dict.get('do_plot_filtered_sig'):
            axes = mp.plot_signal_filtered_each(peaks, segment, fsigs, filters, debug_dict)
            if axes is None:
                debug_dict['do_plot_filtered_sig']=False
                debug_dict['do_plot_filtered_sig_wait']=False

    print('\nfinish filtering')

    json.dump(params,open(params['auto_params']['json_fname'],'w'),indent=2)
    return params,np.array(aline_orders)


def hilbert_all_aline_orders(params, aline_orders, process_type=None, resample=None):
    """
    Create envelope of all orders.
    :param params:
    :param aline_fw_orders:
    :return:
    """
    assert type(process_type) is str, 'Please set process_type to \'fw\' or \'rv\'.'
    print('start hilbert transforming ({})'.format(process_type))
    print(np.shape(aline_orders))

    print('start hilbert: ',np.shape(aline_orders))
    aline_orders_hlb = sg.hilbert(aline_orders, axis=2)
    print('finished hilbert transforming')

    # mp.plot_hilbert(aline_fw_orders_hlb)
    return params, aline_orders_hlb


def compute_rescale_dict(params, num_orders, num_samples):
    # fraction of sample length for each order.
    # The accuracy is not critical so we just force float to int.
    fraction = np.floor(num_samples/num_orders).astype(int)

    # Compute the true axial length of each order in depth using sample units.
    # Sample units can be converted to a physical length based on the digitizer settings, scan range, and wavelength.
    rescale_dict = {}
    for on in range(1,num_orders+1,1):  # on = num_orders ... 1
        rescale_dict[on] = num_samples - fraction * (num_orders - on)  # compute samples for rescaling order

    return rescale_dict

def compute_correlation_adjacent_orders(params, rescale_dict, aline_orders_hilbert):
    """

    :param params:
    :param aline_orders_hilbert:
    :return:
    """
    correlation_dict = {}
    print(np.shape(aline_orders_hilbert))
    num_alines = np.shape(aline_orders_hilbert)[0]
    num_orders = np.shape(aline_orders_hilbert)[1]
    num_samples = np.shape(aline_orders_hilbert)[2]

    # Compute correlation between orders and store the correlation into correlation_dict.
    # The correlation is used to find the spatial alignment between the orders
    # The correlation_dict will hold correlation values for each order but for all A-lines!
    # So it will contain num_order items.
    correlation_dict = {}

    # Using each 10 steps shows nearly no difference compared to using all steps.
    # Exceptions may be if a mirror signal is very bad. But then it seems better
    # to record a better mirror signal.
    steps = params['manual_params']['steps_correlate_orders']
    use_savgol = False # this is not sure if it makes anything better. Test show no difference.
    print('Use savgol',use_savgol)
    print('Align orders by correlation ...')
    print('Using {} step for correlation! (steps_correlate_orders)'.format(steps))
    print('    ',end='') # print some progress number
    for n,al in enumerate(aline_orders_hilbert[::steps]):
        print('\b\b\b\b{: 4d}'.format(n), end='')

        # The order_dict holds each order for this A-line.
        # The order_dict holds the true length for each order and dict is the only way to handle this easily.
        # Re- sample each order to the true spatial length in depth in sample units.
        order_dict = {}
        for ol, on in zip(al, range(1,num_orders+1,1)):  # on = num_orders ... 0
            ol = sg.resample(x=abs(ol), num=rescale_dict[on])
            if use_savgol:
                ol = mp.sg.savgol_filter(x=ol, window_length=151, polyorder=2)
            ol[np.nonzero(ol<=0)]=abs(ol).min() # to avoid errors from fft, logs, etc.
            order_dict[on] = ol

        # Correlate adjacent orders to find array index shift positions and
        # sum for all correlation values for all mirror positions.
        # This correlation can be used to compute the amount the orders need to be shifted spatially to align properly.
        # Take note that if no overlap between orders exist then the correlation is low.
        # Also take note that we reverse the correlation output which is required ...
        for n,m in zip(range(1,num_orders),range(2,num_orders+1)):
            if correlation_dict.get(m) is None:
                correlation_dict[m] = np.correlate(order_dict[n][::-1], order_dict[m][::-1], 'full')
            else:
                correlation_dict[m] += np.correlate(order_dict[n][::-1], order_dict[m][::-1], 'full')
    print('') # final print of progress

    return params, correlation_dict


def compute_order_index_shift(params, num_orders, correlation_dict):
    # Compute the lowest correlation acceptable based on the detected values in the previous step.
    # This allows us to reject correlation artifacts from orders that actually do not overlap (i.e. 1st or 2nd order)
    min_expected_correlation = correlation_dict[num_orders].max()

    order_index_shift_dict = {}
    prominence_fraction_max_peak = 0.9
    print('Use prominence based on max_peak * {}'.format(prominence_fraction_max_peak))
    # Traverse over all correlation values to compute the shift in space for each order and store the shift in
    # order_index_shift_dict.
    # Note that we reverse the correlation graphs "[::-1]" because we need to start with the highest orders to
    # have overlapping regions.
    # Lower orders may not overlap and will have no initial correlation values!
    # TODO: The reverse may not be essential but we can discuss this later.
    for on, ol in list(correlation_dict.items())[::-1]:
        pk,_ = sg.find_peaks(ol,height=min_expected_correlation,prominence=ol.max()*prominence_fraction_max_peak)
        if any(pk):
            # If here are multiple peaks in the correlation, then the signal is corrupt.
            assert len(pk) == 1, 'Peak detection found multiple peaks. Data quality may be insufficient.'
            order_index_shift_dict[on] = pk[0]
        else:
            order_index_shift_dict[on] = None

    # Order 1 can never have any correlation
    order_index_shift_dict[1] = None

    # warn('plot_all_correlation_in_order_index_shift is active here')
    # mp.plot_all_correlation_in_order_index_shift(num_orders, correlation_dict, order_index_shift_dict)

    valid_shift_values = np.array(list(order_index_shift_dict.values()))
    valid_shift_values = valid_shift_values[valid_shift_values.nonzero()] # remove None items
    mean_shift_distance = abs(np.diff(valid_shift_values).mean()).astype(int)
    assert mean_shift_distance is not None, 'Value for mean_shift_distance not available. Data quality may be insufficient.'
    print('shift_lst',valid_shift_values)
    print('mean_shift_distance',mean_shift_distance)

    # Replace missing order_index_shifts using mean_shift_distance from all other orders.
    # In the first loop we obtain the missing shift_index and
    # the second loop moves based on the 1st order shift such that the 1st order is zero.
    # TODO:Double looping over the order_index_shift seems a bit arbitrary, but provides plausible results.
    # Maybe there is a more efficient way.
    for on,shift in order_index_shift_dict.items():
        if shift is None:
            order_index_shift_dict[on] = order_index_shift_dict[on+1] - mean_shift_distance
    o1 = order_index_shift_dict[1]
    # Make that the 1st order has zero shift:
    for on,shift in order_index_shift_dict.items():
        order_index_shift_dict[on] = order_index_shift_dict[on] - o1

    return params, order_index_shift_dict

def compute_padding_for_assemble(params, num_orders, rescale_dict, order_index_shift_dict,process_type):
    assert np.all([idx is not None for idx in order_index_shift_dict.values()]), \
    'Parameter \'order_index_shift_dict\' contains None items.\nCould be a problem with \'compute_order_index_shift\' function.'
    print('shift dict:',order_index_shift_dict)
    # Compute left padding for all orders
    # We need to compute in reverse to have the longest order length available
    left_pad_dict = {}
    right_pad_dict = {}
    correction = params['manual_params']['aline_pad_correction_'+process_type]  # integer only
    print('Padding using correction shift {}'.format(correction))
    for on in range(1,num_orders+1):
        correlation_index = order_index_shift_dict[on]
        left_pad_dict[on] = correlation_index + rescale_dict[on] + on * correction
    for on in range(1,num_orders):
        right_pad_dict[on] = (left_pad_dict[num_orders] + rescale_dict[num_orders]) - (left_pad_dict[on] + rescale_dict[on])
    right_pad_dict[num_orders] = 0

    return params,left_pad_dict,right_pad_dict


def merge_order_in_assemble(params,aline_orders_hilbert, rescale_dict, left_pad_dict, right_pad_dict, is_reversed, debug_dict={}):
    """
    Merge orders into a complete A-line and create a B-frame and plot the B-frame with imshow.
    :param aline_orders_hilbert:
    :param rescale_dict: from function "mf.compute_rescale_dict" a dictionary with the true spatial scale of each order
    :param left_pad_dict: from function "mf.compute_padding_for_assemble" to obtain samples to pad to a complete array
    for one A-line.
    :param right_pad_dict: see left_pad_dict.
    :param is_reversed: True --> flip the orientation of each order
    :return: params, bframe
    """

    num_alines = np.shape(aline_orders_hilbert)[0]
    bframe = None
    # Mean of empty slice can happen at merging the gaps into the image array is not critical
    mp.filterwarnings(action='ignore', category=RuntimeWarning)

    revert_order = is_reversed
    if revert_order: warn('Reverting orders is active')
    print('Start assembling B-frame:')
    print('         ',end='') # print some progress number
    # for al_nr, al in enumerate(aline_orders_hilbert[debug_dict['start_at']::1]):
    for al_nr, al in enumerate(aline_orders_hilbert):
        print('\b\b\b\b\b\b\b\b\b{: 4d}/{:4d}'.format(al_nr,num_alines), end='')

        aline = None
        for i,ol in enumerate(al):
            on = i+1
            pad = left_pad_dict[on]
            ol = (abs(sg.resample(x=abs(ol), num=rescale_dict[on])))
            if revert_order:
                ol = ol[::-1]

            # make padding visible "pad_content=vertical_offset"
            # pad_content = vertical_offset

            pad_content = np.nan
            ol = np.pad(array=ol,pad_width=pad,mode='constant',constant_values=pad_content)[:-pad]
            ol = np.concatenate((ol, np.ones(right_pad_dict[on])*pad_content))

            if aline is None: # create a full length array
                aline = np.ones(len(ol))*np.nan

            merge_method = params['manual_params']['aline_merge_method']
            cut_forward = params['manual_params']['cut_forward']
            if merge_method in 'summing':
                aline = np.nansum((aline, ol), axis=0)
            elif merge_method in 'mean':
                aline = np.nanmean((aline, ol), axis=0)
            elif merge_method in 'cut':
                aline = np.concatenate((aline[:pad+cut_forward*on], ol[pad+cut_forward*on:]))
            elif merge_method in 'correlate':
                raise NotImplementedError()
            else:
                raise Exception('The parameter \'aline_merge_method\' must be \'summing\' or \'mean\'.')

        if bframe is None:
            bframe = aline
        else:
            bframe = np.vstack((bframe, aline))
        # print(np.shape(bframe))
        # print(aline)
        # print into lines to allow zoom during plotting.

    print('') # print final for progress
    print('Assembling B-frame done.')
    bframe[np.where(np.isnan(bframe))] = np.nanmedian(bframe) # fill missing data with median
    # Don't use resample due to generating spurious negative values
    # bframe = sg.resample(x=bframe, num=params['manual_params']['resample_to_vertically'], axis=1)
    steps = np.int(np.round(bframe.shape[1]/params['manual_params']['resample_to_vertically']))
    bframe = bframe[:,::steps]
    print(bframe.shape)
    return params, bframe.T

def assemble_orders_to_alines(params, aline_envelope, is_reversed, process_type = None, debug_dict={}):
    """
    ...
    :param params:
    :param aline_envelope:
    :return:
    """
    assert type(process_type) is str, 'Please set process_type to \'fw\' or \'rv\'.'

    print(np.shape(aline_envelope))
    num_alines = np.shape(aline_envelope)[0]
    num_orders = np.shape(aline_envelope)[1]
    num_samples = np.shape(aline_envelope)[2]

    print('num orders: {}'.format(num_orders))
    print('Compute order index shift positions ...')

    if params['auto_params']['is_calibration']:
        rescale_dict = compute_rescale_dict(params, num_orders, num_samples)
        params, correlation_dict = compute_correlation_adjacent_orders(params, rescale_dict, aline_envelope)
        params, order_index_shift_dict = compute_order_index_shift(params, num_orders, correlation_dict)
        # the list comprehension converts the numbers to normal int for json to work
        params['auto_params']['rescale_'+process_type+'_dict'] = {k:int(v) for k,v in rescale_dict.items()}
        params['auto_params']['order_index_shift_'+process_type+'_dict'] = {k:int(v) for k,v in order_index_shift_dict.items()}
        json.dump(params, open(params['auto_params']['json_fname'], 'w'), indent=2)
    else:
        # need to convert keys back because json can not handle numeric keys
        rescale_dict = {int(k):v for k,v in params['auto_params']['rescale_'+process_type+'_dict'].items()}
        order_index_shift_dict = {int(k):v for k,v in params['auto_params']['order_index_shift_'+process_type+'_dict'].items()}

    params, left_pad_dict, right_pad_dict = compute_padding_for_assemble(params, num_orders, rescale_dict, order_index_shift_dict, process_type)

    if debug_dict.get('plot_each_order'):
        mp.plot_each_order_in_assemble(aline_envelope, rescale_dict, left_pad_dict, right_pad_dict, debug_dict)
    # mp.plot_merge_order_in_assemble(aline_orders_hilbert[::-1], rescale_dict, left_pad_dict, right_pad_dict)
    # mp.plot_merge_order_in_assemble(aline_fw_orders_hlb, rescale_dict, left_pad_dict, right_pad_dict)
    params,bframe = merge_order_in_assemble(params, aline_envelope, rescale_dict, left_pad_dict, right_pad_dict, is_reversed, debug_dict)
    return params, bframe


def compute_median_background(params, bframe, process_type = None):
    background = sg.savgol_filter(np.median(bframe,axis=1,keepdims=False),window_length=101,polyorder=1,axis=0)*0.15
    background = np.array([background])
    return background


def subtract_background(params, bframe, background, process_type = None):
    # global_min = np.min(bframe)
    # back_max = np.max(background)
    bframe = bframe - background  #* global_min / back_max * 1e6
    # all values less than zero are set to a value below the noise floor.
    return bframe

