import numpy as np
import scipy.signal as sg
import scipy.fftpack as spfft
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable # for scaling colorbar

# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes,Line2D
from warnings import warn, filterwarnings
import NEW_CODE_VC.mro_functions as mf
import os

def figure(*args, **kwargs):
    if plt.get_backend() in 'Qt5Agg':
        ret = plt.figure(*args, **kwargs)

        if len(plt.get_fignums()) > 1:
            fig_prev = plt.get_fignums()[-2]
            prev_pos_point = plt.figure(num=fig_prev).canvas.manager.window.pos()
            xp, yp = prev_pos_point.x(), prev_pos_point.y()
            # pos_point = plt.get_current_fig_manager().window.pos()
            # x,y = pos_point.x(),pos_point.y()
            plt.figure(num=ret.number)
            plt.get_current_fig_manager().window.move(xp+20,yp+20)
            return ret
        else:
            plt.get_current_fig_manager().window.move(0,0)
            return ret
    else:
        return plt.figure(*args, **kwargs)

def subplots(*args, **kwargs):
    if plt.get_backend() in 'Qt5Agg':
        ret = plt.subplots(*args, **kwargs)
        if len(plt.get_fignums()) > 1:
            fig_prev = plt.get_fignums()[-2]
            prev_pos_point = plt.figure(num=fig_prev).canvas.manager.window.pos()
            xp, yp = prev_pos_point.x(), prev_pos_point.y()
            # pos_point = plt.get_current_fig_manager().window.pos()
            # x,y = pos_point.x(),pos_point.y()
            plt.figure(num=ret[0].number)
            plt.get_current_fig_manager().window.move(xp+20,yp+20)
            return ret
        else:
            plt.get_current_fig_manager().window.move(0,0)
            return ret
    else:
        return plt.subplots(*args, **kwargs)


def plot_mirror_data_fw_rv(params, data_fw, data_rv, debug_dict):
    for fw, rv in zip(data_fw,data_rv):
        if not hasattr(plot_mirror_data_fw_rv,'fig'):
            fig, (axs1,axs2) = subplots(nrows=1,ncols=2,num='plot_mirror_data_fw_rv')
            plot_mirror_data_fw_rv.fig = fig
            axs1.plot(fw)
            axs2.plot(rv)
            axs1.set_title('FW')
            axs2.set_title('RV')
            fig.suptitle('Non linear')
            fig.subplots_adjust(0.8)
            fig.tight_layout()
        else:
            axs1.cla()
            axs2.cla()
            axs1.set_title('FW')
            axs2.set_title('RV')
            axs1.plot(fw)
            axs2.plot(rv)

        if debug_dict.get('do_stop'):
            plt.draw()
            plt.waitforbuttonpress()
        else:
            plt.draw()
            plt.pause(0.1)

def plot_non_linear_phase(params=None, debug_dict={}):
    phase_fw = np.load(params['manual_params']['phase_fw'])
    phase_rv = np.load(params['manual_params']['phase_rv'])

    if debug_dict.get('do_plot'):
        figure(num="phase with sav_gol fitting")
        plt.plot(phase_fw,label='forward')
        plt.plot(phase_rv,label='reverse')
        plt.ylabel("Angle")
        plt.title("phase with sav_gol fitting")
        plt.legend()
        plt.tight_layout()
        plt.pause(0.1) # to see something even if no stop
        if debug_dict.get('do_stop'):
            plt.draw()
            plt.waitforbuttonpress()
        else:
            plt.pause(0.1)

def plot_linear_signal(params, linear_fw_data, linear_rv_data, debug_dict = {}):

    if debug_dict.get('do_plot'):
        figure(num='Linear Signal')
        button_press.fig = plt.gcf()
        plt.connect('key_press_event', button_press)
        button_press.close = False
        plt.suptitle('Stop with \'x\'')

        for fw_segment,rv_segment in zip(linear_fw_data, linear_rv_data):
            plt.subplot(121)
            plt.plot(rv_segment)
            plt.title("Reverse")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude ( a.u) ")

            plt.subplot(122)
            plt.plot(fw_segment)
            plt.title("Forward")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude ( a.u) ")
            plt.draw()
            plt.tight_layout()
            plt.subplots_adjust(top=0.8)
            is_key = 0 # no mouse clicks
            while plt.waitforbuttonpress() == is_key: pass
            if button_press.close: break
            plt.clf()

def plot_linear_phase(params,linear_fw_data,linear_rv_data,debug_dict={}):
    sig_start = params['manual_params']['sig_start']
    sig_end = params['manual_params']['sig_end']
    sig_step = params['manual_params']['sig_step']
    if debug_dict.get('do_plot'):
        figure(num='Phase')
        linear_fw_sum = sum(linear_fw_data[sig_start:sig_end:sig_step])
        linear_rv_sum = sum(linear_rv_data[sig_start:sig_end:sig_step])
        phase_fw = np.unwrap(np.angle(sg.hilbert(linear_fw_sum)))
        phase_rv = np.unwrap(np.angle(sg.hilbert(linear_rv_sum)))
        plt.plot(phase_fw,label='forward')
        plt.plot(phase_rv,label='reverse')
#        plt.plot(np.abs(phase_fw-phase_rv))
        plt.xlabel("Samples")
        plt.ylabel("Angle")
        plt.title("Linear phase")
        plt.legend()
        plt.tight_layout()
        plt.pause(0.1)
#        plt.show()

def plot_debug_json_freq_update(linear_fw_data,linear_rv_data,debug_dict):
    fig, axs = subplots(nrows=2, ncols=2, figsize=(12,5), num='plot_debug_json_freq_update')
    if not debug_dict.get('start_at'): debug_dict['start_at'] = 0
    for fw, rv in zip(linear_fw_data[debug_dict['start_at']:],linear_rv_data[debug_dict['start_at']:]):
        axs[0,0].plot(fw,lw=0.5)
        axs[0,1].plot(rv[::-1],lw=0.5)
        axs[1,0].semilogy(abs(spfft.fft(fw)),lw=0.5)
        axs[1,0].set_xlim((0,1500))
        axs[1,1].semilogy(abs(spfft.fft(rv)),lw=0.5)
        axs[1,1].set_xlim((0,1500))
        plt.waitforbuttonpress()
        [a.cla() for a in axs.flatten()]

def plot_linear_signal_and_do_fft(params, linear_fw_data, linear_rv_data, debug_dict):
    '''
    Show build up of sum of the FFT signal
    :param do_plot:
    :param params:
    :param linear_fw_data:
    :param linear_rv_data:
    :param debug_dict: {'do_plot':BOOL, 'start_at':int}
    :return:
    '''
    buffer_time = params['manual_params']['buffer_time']

    f_fw = spfft.fftshift(spfft.fftfreq(n=linear_fw_data.shape[1], d=buffer_time / linear_fw_data.shape[1]))
    f_rv = spfft.fftshift(spfft.fftfreq(n=linear_rv_data.shape[1], d=buffer_time / linear_rv_data.shape[1]))

    if debug_dict.get('do_plot') and not hasattr(plot_linear_signal_and_do_fft,'fig'):
        fig = figure(num='Linear Signal and FFT')
        button_press.fig = fig
        plt.connect('key_press_event', button_press)
        button_press.close = False

    fft_fw_all_steps = []
    fft_rv_all_steps = []
    if not debug_dict.get('start_at'): debug_dict['start_at']=50
    if debug_dict.get('do_plot'):
        for fw_sig, rv_sig in zip(linear_fw_data[debug_dict['start_at']:], linear_rv_data[debug_dict['start_at']:]):
            fft_fw_sig = abs(spfft.fftshift(spfft.fft(fw_sig)))
            fft_fw_all_steps.append(fft_fw_sig)
            fft_rv_sig = abs(spfft.fftshift(spfft.fft(rv_sig)))
            fft_rv_all_steps.append(fft_rv_sig)

            plt.subplot(221)
            plt.plot( rv_sig)
            plt.title("Reverse")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude ( a.u) ")

            plt.subplot(222)
            plt.plot(fw_sig) # plot the phase
            plt.title("Forward")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude ( a.u) ")

            plt.subplot(223)
            plt.semilogy(f_fw*1e-3, fft_fw_sig)
            plt.xlim((0, f_fw.max()*1e-3/12))
            plt.title("fft_of_one_mirror_step forward")
            plt.xlabel("frequency (kHz)")
            plt.ylabel("Amplitude (a.u)")

            plt.subplot(224)
            plt.semilogy(f_rv*1e-3, fft_rv_sig)
            plt.xlim((0, f_rv.max()*1e-3/12))
            plt.title("fft_of_one_mirror_step reverse")
            plt.xlabel("frequency (kHz)")
            plt.ylabel("Amplitude (a.u)")

            plt.suptitle('Press key to see signal steps\n(Press \'x\' to skip)')
            plt.tight_layout()
            plt.subplots_adjust(top=0.8)
            plt.draw()
            is_key = 0
            while plt.waitforbuttonpress() == is_key: pass
            if button_press.close: break
            plt.clf()

    return(fft_fw_all_steps)

def plot_align_img_data_debug(params,data):
    data_samples = data.shape[1]
    img_cut_off_top = 0
    img_cut_off_end = 0

    img_cut_off_tops = [5500] #list(range(0,15000,100))
    img_cut_off_ends = [330] #np.zeros(15000//100).astype(int)

    for img_cut_off_top, img_cut_off_end in zip(img_cut_off_tops, img_cut_off_ends):
        flat_len = np.prod(data.shape)
        data = np.reshape(data, flat_len)  # make it 1D (seems to avoid copy)
        data = data[img_cut_off_top:]  # discard top / beginning to move all orders
        segments_rem = data.shape[0] // data_samples  # compute remaining orders due to shorter total length
        samples_tail = data.shape[0] - segments_rem * data_samples  # number of samples of last shorter data tail
        data = data[: data.shape[0] - samples_tail]  # remove last incomplete data tail
        data = data[: data.shape[0] - img_cut_off_end * segments_rem]  # for shortening each order by img_cut_off_end remove those samples as well
        data = np.reshape(data, [segments_rem, data_samples - img_cut_off_end])  # reshape back
        new_data_len = data.shape[1]
        data_aligned_fw = data[:, 0: new_data_len // 2]
        data_aligned_rv = data[:, new_data_len // 2:new_data_len]

        if not hasattr(plot_align_img_data_debug, 'fig'):
            fig, axs = subplots(nrows=3, ncols=2, figsize=(12, 6))
            plot_align_img_data_debug.fig = fig
            button_press.fig = fig
            plt.connect('key_press_event', button_press)
            button_press.close = False

        show_segment_from = 0
        show_segment_to = 200
        show_segment_step = 50
        rr = range(show_segment_from,show_segment_to,show_segment_step)

        by_sum = False
        if by_sum:

            axs[0, 0].cla()
            axs[1, 0].cla()
            axs[2, 0].cla()
            axs[0, 1].cla()
            axs[1, 1].cla()
            axs[2, 1].cla()

            d = data
            fw = data_aligned_fw
            rv = data_aligned_rv
            axs[0, 0].plot(d[100], lw=0.5), axs[0, 0].set_ylim((30000, -30000))
            axs[1, 0].plot(fw[100], lw=0.5), axs[1, 0].set_ylim((30000, -30000))
            axs[2, 0].plot(rv[100], lw=0.5), axs[2, 0].set_ylim((30000, -30000))
            axs[1, 1].plot(np.sum((abs(fw)),axis=0), lw=0.5), axs[1, 1].set_xlim((0, 1500))
            axs[2, 1].plot(np.sum((abs(rv)),axis=0), lw=0.5), axs[2, 1].set_xlim((0, 1500))

            fig.suptitle('img_cut_off_top: {}'.format(img_cut_off_top))
            plt.tight_layout()
            fig.subplots_adjust(top=0.88)


        by_step = True
        if by_step:
            for d, fw, rv in zip(data[rr], data_aligned_fw[rr], data_aligned_rv[rr]):
                axs[0,0].cla()
                axs[1,0].cla()
                axs[2,0].cla()
                axs[0,1].cla()
                axs[1,1].cla()
                axs[2,1].cla()

                axs[0,0].plot(d,  lw=0.5),axs[0,0].set_ylim((30000, -30000))
                axs[1,0].plot(fw, lw=0.5),axs[1,0].set_ylim((30000, -30000))
                axs[2,0].plot(rv, lw=0.5),axs[2,0].set_ylim((30000, -30000))
                axs[0,1].semilogy(abs(spfft.fft(d )),lw=0.5),axs[0,1].set_xlim((0,1500))
                axs[1,1].semilogy(abs(spfft.fft(fw)),lw=0.5),axs[1,1].set_xlim((0,1500))
                axs[2,1].semilogy(abs(spfft.fft(rv)),lw=0.5),axs[2,1].set_xlim((0,1500))

                fig.suptitle('img_cut_off_top: {}'.format(img_cut_off_top))
                plt.tight_layout()
                fig.subplots_adjust(top=0.88)
                plt.draw()

                plt.waitforbuttonpress()
                # if button_press.close: break
                # man_pars.plt.pause(0.1)


        plt.waitforbuttonpress()
        if button_press.close: break
    assert False, 'stop here.'


def plot_fft_sum_and_peaks(params, debug_dict):
    # fft_fw_sum = np.load(params['manual_params']['fft_fw_sum'])
    fft_fw_sum_svg = np.load(params['manual_params']['fft_fw_sum_svg'])
    peaks_fw = np.fromstring(params['auto_params']['peaks_fw_px'].strip('[]'),sep=',').astype(np.int)
    # fft_rv_sum = np.load(params['manual_params']['fft_rv_sum'])
    fft_rv_sum_svg = np.load(params['manual_params']['fft_rv_sum_svg'])
    peaks_rv = np.fromstring(params['auto_params']['peaks_rv_px'].strip('[]'),sep=',').astype(np.int)

    if params['auto_params'].get('warn_peaks_array') or params['auto_params'].get('warn_peaks_freq_diff'):
        debug_dict={'do_plot':True, 'do_wait':True}

    if debug_dict.get('do_plot'):
        figure(num='FFT summation')
        plt.subplot(121)
        plt.semilogy(fft_fw_sum_svg)
        plt.semilogy(peaks_fw, fft_fw_sum_svg[peaks_fw],"x")
        # plt.semilogy(f_new,fft_sum_new,f_new[peaks_fw],fft_sum_new[peaks_fw],"x")
        # plt.semilogy([f_new[0],f_new[-1]],[np.median(fft_sum_new),np.median(fft_sum_new)])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (a.u)")
        plt.title("fft_summation and\ndetection of peaks_fw (raw) {}".format(len(peaks_fw)))
        len_tot_px = fft_fw_sum_svg.shape[0]
        plt.xlim((0, peaks_fw.max()*1.5))
        plt.subplot(122)
        plt.semilogy(fft_rv_sum_svg)
        plt.semilogy(peaks_rv, fft_rv_sum_svg[peaks_rv], "x")
        # plt.semilogy(f_new,fft_sum_new,f_new[peaks_rv],fft_sum_new[peaks_rv],"x")
        # plt.semilogy([f_new[0],f_new[-1]],[np.median(fft_sum_new),np.median(fft_sum_new)])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (a.u)")
        plt.title("fft_summation and\ndetection of peaks_rv (raw) {}".format(len(peaks_rv)))
        len_tot_px = fft_rv_sum_svg.shape[0]
        plt.xlim((0, peaks_rv.max() * 1.5))

        plt.suptitle(params['auto_params']['time_stamp']+'\n'+params['auto_params']['title_str'])
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)

        if debug_dict['do_wait']:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.1)
        # plt.show()


def plot_apodization(params, linear_data):
    segment_len  = linear_data.shape[1]

    do_plot = False
    if do_plot:
        start_at = 0
        if not hasattr(plot_apodization,'fig'):
            fig, axs = figure(ncols=2, nrows=1)
            plot_apodization.fig = fig

        for segment in linear_data[start_at:]:
            axs[0].plot( segment )

            segment = segment * sg.windows.tukey(M=segment_len, alpha=params['manual_params']['alpha'])

            axs[1].plot( segment )
            # mp.Axes.get_y
            axs[1].set_ylim(axs[0].get_ylim())
            plt.draw()
            plt.waitforbuttonpress()
            axs[0].cla()
            axs[1].cla()



def plot_filter_response_each(peaks, h, debug_dict):
    """
    Single plot to show filter response for each order.
    :param h: the filter response of one filter for one order
    :return: axes of the filter response plot
    """
    if debug_dict['show_each_filter']:
        # I am using a function attribute 'figure_created' as a convenient way to recall if I have a figure visible or created already.
        # This has the advantage that this is local to this function but persistent between function calls, because
        # the attribute is kept like the function help string - even available if the function is not called.
        if not hasattr(plot_filter_response_each,'figure_created'):
            plot_filter_response_each.figure_created = True
            fig, axes = subplots(nrows=1,ncols=1,num='Filter response')
            plot_filter_response_each.axes = axes
            plot_filter_response_each.fig = fig
        else:
            fig = plot_filter_response_each.fig
            axes = plot_filter_response_each.axes

        axes.set_title('Filter response and FFT sig')
        # from matplotlib.pyplot import Axes
        # Axes.plot()
        axes.semilogy(abs(spfft.fftshift(h)))
        axes.set_xlim((len(h)/2, len(h)/2+peaks.max()*1.25))
        axes.set_ylim((1e-7,abs(h).max()*1.5 ))

        plt.tight_layout()
        plt.draw()
        if debug_dict['wait_each_filter']:  plt.waitforbuttonpress()
    else:
        fig = None
        axes = None

    return fig, axes



def plot_filter_response_vs_signal_in(peaks, filters, linear_signal, debug_dict):
    """
    Show input signal FFT vs filter response.
    Allow stepping through each input signal (linear_signal).
    :param filter_response:
    :param linear_signal:
    :param do_plot:
    :param do_wait:
    :return: Axes
    """

    sig_len = linear_signal.shape[1]
    filter_response = mf.make_filter_response_array(peaks, filters, sig_len, debug_dict)

    if debug_dict['do_plot_filter_and_sig']:
        debug_dict['show_each_filter'] = True
        for fr in filter_response: # plot all filters
            fig,axs = plot_filter_response_each(peaks, fr['H'], debug_dict)

        # Prepare a signal plot and keep
        aline_len_idx = 1
        last_plot = -1
        # add one plot-line for the input signal
        input_signal_plot = axs.semilogy(np.ones(np.shape(linear_signal)[aline_len_idx]),'k',lw=1)[last_plot]

        # Step through a-line signals FFT
        for aline in linear_signal[debug_dict['start_at']:]:
            fft_sig = abs(spfft.fftshift(spfft.fft(aline)))
            mxv = max(fft_sig) # match signal height with filter response
            fft_sig /= mxv     # match signal height with filter response
            input_signal_plot.set_ydata(fft_sig)
            plt.draw()
            if debug_dict['do_plot_filter_and_sig_wait']: plt.waitforbuttonpress()
    else:
        fig = None
        axs = None

    return fig,axs


def plot_signal_filtered_each(peaks, segment, fsigs, filters, debug_dict):
    """
    Plot filter response again but also the resulting filtered signal (using plot layout 2x2)
    :param sig:
    :param fsig:
    :param filter_response:
    :param linear_signal:
    :param do_plot:
    :param do_wait:
    :return:
    """
    if debug_dict['do_plot_filtered_sig']:
        debug_dict['show_each_filter'] = True
        # sig_len = fsigs.shape[2]
        sig_len = segment.shape[0]
        filter_response = mf.make_filter_response_array(peaks, filters, sig_len, debug_dict)
        # I am using a function attribute 'figure_created' as a convenient way to recall if I have a figure visible or created already.
        # This has the advantage that this is local to this function but persistent between function calls, because
        # the attribute is kept like the function help string - even available if the function is not called.
        if not hasattr(plot_signal_filtered_each,'figure_created'):
            plot_signal_filtered_each.figure_created = True
            fig,axes = subplots(nrows=2,ncols=2,num='Filter response and signals')
            plot_signal_filtered_each.axes = axes
            plot_signal_filtered_each.fig = fig
            button_press.fig = fig
            plt.connect('key_press_event', button_press)
            button_press.close = False

        else:
            axes = plot_signal_filtered_each.axes
            fig = plot_signal_filtered_each.fig

        # Create filter response data only if not yet copied
        if not any(axes[0,0].get_lines()):
            # copy plot data
            for fr in filter_response: # plot all filters
                figh,axs = plot_filter_response_each(peaks, fr['H'], debug_dict)
            plt.close(figh) # close temp figure
            # paste plot data
            for l in axs.get_lines():
                axes[0,0].add_line(Line2D(xdata=l.get_xdata(),ydata=l.get_ydata(),color='k',linewidth=0.5))
            axes[0,0].set_title('Filter vs input signal FFT')
            axes[0,0].autoscale(enable=True)
            axes[0,0].set_yscale('log')
            h1 = axs.get_lines()[0].get_ydata()
            axes[0,0].set_xlim((len(h1)/2, len(h1)/2 + peaks.max()*0.55))
            axes[0,0].set_ylim((1e-7,1e3))
            # add one dummy plot for signal
            axes[0,0].add_line(Line2D(xdata=axs.get_lines()[0].get_xdata(), ydata=np.ones(len(h1)),color='r',linewidth=0.5))

        fft_sig = abs(spfft.fftshift(spfft.fft(segment)))
        mxv = max(fft_sig) # match signal height with filter response
        fft_sig /= mxv     # match signal height with filter response
        axes[0,0].get_lines()[-1].set_ydata(fft_sig)

        axes[0,1].set_title('Filtered signal')
        if not np.any(axes[0,1].get_lines()):
            for fsig in fsigs:
                axes[0,1].plot(fsig,linewidth=0.5)
        else:
            for fsig in fsigs:
                axes[0,1].get_lines()[0].set_ydata(fsig)

        def show_signal_plain(fsig_processed):
            axes[1, 1].set_title('Filtered signal plain')
            axes[1, 1].plot(fsig_processed,linewidth=0.5)

        def show_signal_FFT(fsig_processed):
            axes[1, 1].set_title('Filtered signal FFT')
            fsig_fft = abs(spfft.fftshift(spfft.fft(fsig_processed)))
            axes[1, 1].semilogy(fsig_fft,linewidth=0.5)
            axes[1, 1].set_xlim((len(fsig) / 2, len(fsig) / 1.8))
            axes[1, 1].set_ylim((1e2, 1e9))

        def show_signal_Hilb(fsig_processed):
            axes[1, 1].set_title('Filtered signal Hilbert')
            fsig_h = abs(sg.hilbert(fsig_processed))
            axes[1, 1].semilogy(fsig_h,linewidth=0.5)
            axes[1, 1].set_ylim((1e1, 1e6 ))

        for fsig in fsigs:
            # show_signal_plain(fsig)
            # show_signal_FFT(fsig)
            show_signal_Hilb(fsig)


        axes[1,0].set_title('Input signal')
        if not np.any(axes[1, 0].get_lines()):
            axes[1, 0].plot(segment, linewidth=0.5)
        else:
            axes[1, 0].get_lines()[0].set_ydata(segment)

        plt.draw()
        plt.tight_layout()
        if debug_dict['do_plot_filtered_sig_wait']:
            plt.waitforbuttonpress()
            if button_press.close:
                button_press.close = False
                return None
        else:
            plt.pause(0.1)

        axes[0,1].cla()
        axes[1,0].cla()
        axes[1,1].cla()
    else:
        fig = None
        axes = None

    return axes


def plot_hilbert( aline_orders_hilbert):
    """
    This plot is really just to check if hilbert works.
    It is not supposed to give a full analysis view.
    Probably it is easier to call the plot directly with the line of code used here.
    :param aline_orders_hilbert:
    :return:
    """
    plt.plot(abs(aline_orders_hilbert[10,0,:]))

def button_press(evt):
    """
    A function that captures a key_press_event 'q' or 'x' allowing to stop waitforbuttonpress.
    In your plotting function add after creation of the figure the line:
    "plt.connect('key_press_event', button_press)"
    and also attach the figure as attrbitue like
    "button_press.fig = fig"

    If you have many figures then you need to use different attributes like "fig1", "fig2", etc.
    :param evt:
    :return:
    """
    print(evt.name)
    # print(evt.press)
    print(evt.x)
    print(evt.y)
    print(evt.guiEvent)
    print(evt.key)
    if evt.key in 'qx':
        # plt.close(button_press.fig)
        button_press.close = True

def plot_raw_aline_orders_hilbert(params, aline_orders_hilbert):
    """
    Plot the raw A-line envelopes after Hilbert transformation.
    Change al_nr to start at another A-line
    :param aline_orders_hilbert:
    :return:
    """
    al_nr = 300
    for al in aline_orders_hilbert[al_nr:]:
        # print(np.shape(al))
        if not hasattr(plot_raw_aline_orders_hilbert,'fig'):
            fig,axs = subplots(nrows=np.shape(al)[0],ncols=1,figsize=(8,7.5),num='Raw unscaled orders after filtering')
            button_press.fig = fig
            plt.connect('key_press_event', button_press)
            plot_raw_aline_orders_hilbert.fig = fig

        # print('num orders: {}'.format(len(al)))

        axs[0].set_title('Press x to stop. #A-line: {: 3d}'.format(al_nr))
        al_nr +=1
        for ol,ax in zip(al,axs):
            if hasattr(button_press,'close'): return
            ax.plot(abs(ol))
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_ylim((1e-1,22000))
        plt.draw()
        # mp.plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0.01, right=1.0, top=0.95 , wspace=None, hspace=None)
        plt.waitforbuttonpress()
        [ax.cla() for ax in axs]
    return params


def plot_orders_scaled(params, aline_orders_hilbert):
    """
    This function is investigating the resampling for each order according to the true spatial depth range.
    This function is not essential for processing.
    The principle is that each subsequent order is N times the depth range than the first one
    see Dsouza 2016 (3.18).
    Here in the code we reduce the samples from the highest order down to the first order.
    Because the lowest order has the least imaging depth and therefore requires the least samples.

    You can see the loop traversing over each A-line (for al in aline_orders_hilbert).
    Each A-line contains all orders (ol) detected.
    The variable 'fraction' is used to reduce / resample each order.
    The variable 'fraction' is simply the maximum samples divided by the number of orders.
    Please take note that we do not make any effort about accurate rounding of the samples and just assume type integer
    for 'fraction'. This is sufficient as the image pixel accuracy is of less relevance.

    The code line

    smpls = num_samples - fraction * on

    computes the new samples (smpls) for each order (on), where 'on' is the integer order number from N-1..0.
    That means that the first order will have the smallest number of samples while the highest order 'N' will have the
    maximum number of samples.


    R. I. Dsouza, “Towards Low Cost Multiple Reference Optical Coherence Tomography for in Vivo and NDT Applications,” Thesis, 2016.
    :param params:
    :param aline_orders_hilbert:
    :return:
    """
    bframe = []
    num_orders = np.shape(aline_orders_hilbert)[1]
    num_samples = np.shape(aline_orders_hilbert)[2]
    fraction = np.floor(num_samples/num_orders).astype(int)
    fig = None
    print('num orders: {}'.format(num_orders))

    for al in aline_orders_hilbert[200:]:
        aline = []
        # print(np.shape(al))
        if fig is None:
            # fig,axs = mp.subplots(nrows=np.shape(al)[0],ncols=1,figsize=(8,7.5))
            fig,ax = subplots(nrows=1, ncols=1)

        for ol,on in zip(al,range(num_orders-1,-1,-1)): # on = num_orders ... 0
            smpls = num_samples - fraction*on
            print(smpls,end=', ')

            # multiply by some number to shift each order a bit up for visual presentation.
            ol = sg.resample(x=abs(ol),num=smpls) + on*1000
            # ol = mf.np.log10(mf.sg.resample(x=abs(ol),num=smpls)) + on*3
            ax.plot(ol)
            ax.set_xlabel('Samples')
            ax.set_ylabel('Intensity (arb.)')
        print('')

        ax.set_ylim((-800,16000)) # for linear scale the lim is matching for the twinx!
        if not hasattr(plot_orders_scaled,'has_y2'): # add twinx only once
            plot_orders_scaled.has_y2 = True
            ax2 = ax.twinx()
            ax2.set_ylabel('Order')
            ax2.set_yticks(np.linspace(0,12.4,14)) # position ticks
            ax2.set_yticklabels(range(0,14))
            ax2.set_ylim((0.3,16))
            ax2.grid(axis='y')

        plt.draw()
        plt.tight_layout()
        plt.waitforbuttonpress()
        ax.cla()


    return params


def plot_summing_of_correlation_values(params, aline_orders_hilbert):
    """
    This function studies the correlation between orders and their use for order alignment.
    This function is not essential for processing.
    Using correlation is not a major discovery but novel in comparison of the previous code.

    Although, by theory we can calculate the position of orders exactly this appears to be not true in practise due to
    minor deviations.
    The method starts with the highest orders and works backwards trying to find the best match of the mirror signal.
    The position parameters are then stored for aligning orders for signals with random scattering properties.

    The number to shift an order array vs the next one is determined by the spacing of the PM and SRM.
    However, this value can not exactly be measured.

    The principle is that each subsequent order is N times the depth range than the first one
    see Dsouza 2016 (3.18).
    Here in the code we reduce the samples from the highest order down to the first order.
    Because the lowest order has the least imaging depth and therefore requires the least samples.

    You can see the loop traversing over each A-line (for al in aline_orders_hilbert).
    Each A-line contains all orders (ol) detected.
    The variable 'fraction' is used to reduce / resample each order.
    The variable 'fraction' is simply the maximum samples divided by the number of orders.
    Please take note that we do not make any effort about accurate rounding of the samples and just assume type integer
    for 'fraction'. This is sufficient as the image pixel accuracy is of less relevance.

    The code line

    smpls = num_samples - fraction * on

    computes the new samples (smpls) for each order (on), where 'on' is the integer order number from N-1..0.
    That means that the first order will have the smallest number of samples while the highest order 'N' will have the
    maximum number of samples.


    R. I. Dsouza, “Towards Low Cost Multiple Reference Optical Coherence Tomography for in Vivo and NDT Applications,” Thesis, 2016.
    :param params:
    :param aline_orders_hilbert:
    :return:
    """
    print(np.shape(aline_orders_hilbert))
    bframe = []
    correlation_dict = {}
    num_orders = np.shape(aline_orders_hilbert)[1]
    num_samples = np.shape(aline_orders_hilbert)[2]
    fraction = np.floor(num_samples/num_orders).astype(int)
    fig = None
    print('num orders: {}'.format(num_orders))

    start_at = 420
    for n,al in enumerate(aline_orders_hilbert[10::1]):
        n = start_at-n
        aline = []
        # print(np.shape(al))
        if fig is None:
            # fig,axs = mp.subplots(nrows=np.shape(al)[0],ncols=1,figsize=(8,7.5))
            fig,axs = subplots(nrows=1, ncols=2, figsize=(10,5))
        axs[0].set_title('A-line #{}'.format(n))
        axs[1].set_title('Build up of correlation graph\n (each order different color)')

        # for ol,on in zip(al,range(0,num_orders,1)): # on = num_orders ... 0
        order_dict = {}
        for ol, on in zip(al[::-1], range(0,num_orders,1)):  # on = num_orders ... 0
            smpls = num_samples - fraction*on

            # multiply by some number to shift each order a bit up for visual presentation.
            # ol = mf.sg.resample(x=abs(ol),num=smpls) + on*1000

            ol = sg.resample(x=abs(ol), num=smpls)
            ol = sg.savgol_filter(x=ol, window_length=151, polyorder=2)
            ol[np.nonzero(ol<=0)]=abs(ol).min() # to avoid errors from fft, logs, etc.
            order_dict[num_orders-on] = ol
            ol = np.log10(ol) + on*3
            axs[0].plot(ol)
            axs[0].set_xlabel('Samples')
            axs[0].set_ylabel('Intensity (arb.)')
        lns = axs[0].get_lines()
        cols = [ln.get_color() for ln in lns]
            # mp.plt.draw()
            # mp.plt.waitforbuttonpress()

        # ax.set_ylim((-800,16000)) # for linear scale the lim is matching for the twinx!
        if not hasattr(plot_summing_of_correlation_values,'has_y2'):
            plot_summing_of_correlation_values.has_y2 = True
            ax2 = axs[0].twinx()
            ax2.set_ylabel('Order')
            ax2.set_yticks(np.linspace(0,num_orders+2,num_orders+1)) # position ticks
            ax2.set_yticklabels(range(num_orders+1,0,-1))
            ax2.set_ylim((0.3,num_orders+3))
            ax2.grid(axis='y')

        # Summing all correlations between orders.
        # In theory each overlapping orders should correlate at the same displacement between the orders independently
        # of the sample mirror displacement.
        # The peak of the correlation should correspond to the pixel displacement to bring the overlapping order to
        # optimal alignemnt.
        # The estimation error in this case is of minimal relevance because it would only present an error of the image
        # layer within the image.
        # But often if the image layer is narrow the exact position of some detected structure is of little relevance.
        # This may be different if this is desired to be a depth measurement. But even then the refractive index would
        # need to be known with sufficient accuracy which is often not the case anyway.
        for n,m in zip(range(1,num_orders),range(2,num_orders+1)):
            if correlation_dict.get(n) is None:
                correlation_dict[n] = np.correlate(order_dict[n][::-1], order_dict[m][::-1], 'full')
            else:
                correlation_dict[n] += np.correlate(order_dict[n][::-1], order_dict[m][::-1], 'full')

        def plot_correlation_build_up():
            """
                Show building up the sum of correlation for each A-line
                Enable waitforbuttonpress to show for each mirror step
            """
            for n, m, c in zip(range(1, num_orders), range(2, num_orders+1), cols[num_orders::-1]):
                # axs[1].plot(mp.np.correlate(order_dict[n][::-1],order_dict[m][::-1],'full') ,color=c )
                axs[1].plot(correlation_dict[n], color=c)
                # mp.plt.draw()
                # mp.plt.waitforbuttonpress()
        plot_correlation_build_up()

        def plot_for_each_mirror_position():
            # mp.plt.pause(0.1) # draw without wait
            # mp.plt.tight_layout()
            plt.draw() # needed for waitforbuttonpress to draw
            plt.waitforbuttonpress() # show all orders processed for each mirror position
            axs[0].cla()
            axs[1].cla()
        plot_for_each_mirror_position()

    def plot_final_sum_of_correlations():
        for n, m, c in zip(range(1, num_orders), range(2, num_orders+1), cols[num_orders::-1]):
            axs[1].plot(correlation_dict[n], color=c)
    # plot_final_sum_of_correlations()

    plt.show()
    return params


def plot_orders_spatially_aligned(params, aline_orders_hlb, is_reversed, process_type, debug_dict={}):
    """
    This function studies the spatial alignment of orders using order_index_shift_dict.
    This function is not essential for processing.

    Each order in this function is aligned using correlation between adjacent orders.
    The correlation values or the correlation graph has a peak if matching mirror signals are available.
    The peak position is stored in order_index_shift_dict for each order.

    This function uses correlation and peak detection to find the order_index_shift_dict.
    Although, by theory we can calculate the position of orders exactly this appears to be not true in practise due to
    minor deviations.
    The method starts with the highest orders and works backwards trying to find the best match of the mirror signal.
    The position parameters are then stored for aligning orders for signals with random scattering properties.

    The number to shift an order array vs the next one is determined by the spacing of the PM and SRM.
    However, this value can not exactly be measured.

    The principle is that each subsequent order is N times the depth range than the first one
    see Dsouza 2016  (3.18).
    Here in the code we reduce the samples from the highest order down to the first order.
    Because the lowest order has the least imaging depth and therefore requires the least samples.

    You can see the loop traversing over each A-line (for al in aline_orders_hilbert).
    Each A-line contains all orders (ol) detected.
    The variable 'fraction' is used to reduce / resample each order.
    The variable 'fraction' is simply the maximum samples divided by the number of orders.
    Please take note that we do not make any effort about accurate rounding of the samples and just assume type integer
    for 'fraction'. This is sufficient as the image pixel accuracy is of less relevance.

    The code line

    smpls = num_samples - fraction * on

    computes the new samples (smpls) for each order (on), where 'on' is the integer order number from N-1..0.
    That means that the first order will have the smallest number of samples while the highest order 'N' will have the
    maximum number of samples.


    R. I. Dsouza, “Towards Low Cost Multiple Reference Optical Coherence Tomography for in Vivo and NDT Applications,” Thesis, 2016.
    :param params:
    :param aline_orders_hlb:
    :return:
    """

    revert_order = is_reversed

    aline_orders_envelope = np.abs(aline_orders_hlb)

    print(np.shape(aline_orders_envelope))
    num_alines = np.shape(aline_orders_envelope)[0]
    num_orders = np.shape(aline_orders_envelope)[1]
    num_samples = np.shape(aline_orders_envelope)[2]
    fraction = np.floor(num_samples/num_orders).astype(int)

    print('num orders: {}'.format(num_orders))
    print('Compute order index shift positions ...')
    print('    ',end='') #print some progress number

    # Compute correlation between orders and store the correlation into correlation_dict.
    # The correlation is used to find the spatial alignment between the orders
    correlation_dict = {}
    # Using each 10 steps shows nearly no difference compared to using all steps.
    # Exceptions may be if a mirror signal is very bad. But then it seems better
    # to record a better mirror signal.
    steps = 10
    print('Using only each {} step for correlation!'.format(steps))
    fig = None

    if fig is None and debug_dict.get('show_correlation') or debug_dict.get('show_all_orders'):
        fig, axs = subplots(nrows=1, ncols=2, figsize=(10, 5), num='in plot_orders_spatially_aligned')
        button_press.fig = fig
        plt.connect('key_press_event', button_press)
        button_press.close = False

    for n,al in enumerate(aline_orders_envelope[::steps]):
        print('\b\b\b\b{: 4d}'.format(n), end='')

        axs[0].cla()
        axs[1].cla()
        axs[0].set_title('Correlation for all A-lines.\nStop with \'x\'')
        axs[1].set_title('Visual placement of orders.')
        plt.suptitle('Orders before merge')
        plt.subplots_adjust(top=0.8)

        # The order_dict addresses each order in one A-line as 'order_dict[ on ]' where 'on' is the order number.

        # TODO: The resale_dict is computed for each A-line for analysis. But this step is not too slow anyway. (Low Prio)
        # The rescale_dict should be valid for all A-lines but we collect it here for each A-line for analytical
        # purpose. At the moment it does not consume too much processing.

        # Is the use of dictionaries effecient?
        # Should we use arrays?
        # First of all arrays MIGHT be more efficient but it is more difficult to analyze the logic.
        # So it would cause more design time and not help for data analysis.
        # Then this is only for calibration line analysis and the slow down may be acceptable at the moment.
        # Imaging the pre-computed values are used anyway.
        # So the processing speed at this stage is not as important at all.
        order_dict = {}
        rescale_dict = {}
        for ol, on in zip(al, range(1,num_orders+1,1)):  # on = num_orders ... 0
            smpls = num_samples - fraction*(num_orders-on) # compute samples for rescaling order
            rescale_dict[on] = smpls
            ol = sg.resample(x=abs(ol), num=smpls)
            # ol = mp.sg.savgol_filter(x=ol, window_length=151, polyorder=2)
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

    # Compute the lowest correlation acceptable based on the detected values in the previous step.
    # This allows us to reject correlation artifacts from orders that actually do not overlap (i.e. 1st or 2nd order)
    min_expected_correlation = correlation_dict[num_orders].max()
    # print(min_expected_correlation)


    order_index_shift_dict = {}
    shift_lst = [] # for computing the mean_shift_distance in order_index_shift_dict
    mean_shift_distance = None
    # Traverse over all correlation values to compute the shift in space for each order and store the shift in
    # order_index_shift_dict.
    # Note that we reverse the correlation graphs "[::-1]" because we need to start with the highest orders to
    # have overlapping regions.
    # Lower orders may not overlap and will have no initial correlation values!
    # TODO: Is reversing correlation_dict essential? Does not change performance. (Prio Low)
    for on, ol in list(correlation_dict.items())[::-1]:
        pk,_ = sg.find_peaks(ol,height=min_expected_correlation,prominence=ol.max()*0.9)
        if any(pk):
            # If here are multiple peaks in the correlation, then the signal is corrupt.
            assert len(pk) == 1, 'Peak detection found multiple peaks. Data quality may be insufficient.'
            order_index_shift_dict[on] = pk[0]
            shift_lst.append(pk[0])
        else:
            # The index shift can be negative if the orders have a gap!
            # mean_shift_distance = mf.np.mean(abs(mf.np.diff(shift_lst))).astype(int)
            # order_index_shift_dict[on] = order_index_shift_dict[on+1] - mean_shift_distance
            order_index_shift_dict[on] = None
    mean_shift_distance = abs(np.diff(shift_lst).mean()).astype(int)
    assert mean_shift_distance is not None, 'Value for mean_shift_distance not available. Data quality may be insufficient.'
    # Order 1 can never have any correlation
    order_index_shift_dict[1] = None

    # Store the number of orders that do not have overlap.
    # Because for those orders we have to add pre-computed padding space to create the full A-line array.
    none_number_of_elements = 0

    # plot all correlations and peaks
    # Please not that we ignore the order 1 as it has no correlation values (range(num_orders, 1, -1))!
    # We plot starting with the highest orders as those will have correlation values (reversed range).
    if debug_dict.get('show_correlation'):
        for on in range(num_orders,1,-1):
            corr_graph = correlation_dict[on]
            shift_point = order_index_shift_dict[on]
            axs[0].plot(corr_graph)
            # negative values are gaps; but here we ignore them for the moment as we don't have correlation data.
            # so where no peak is we can't plot it.
            if shift_point is not None:
                axs[0].plot(shift_point,corr_graph[shift_point],'rx')
            else:
                none_number_of_elements += 1
                warn('Order {} is ignored. No order_index_shift available.'.format(on))

            if debug_dict.get('do_stop'):
                plt.draw()
                plt.waitforbuttonpress()
                if button_press.close: break
            # else:
            #     plt.pause(0.1)

    print('\nshift_lst',shift_lst)
    print('mean_shift_distance',mean_shift_distance)
    print('shift dict:',order_index_shift_dict)

    # Replace missing order_index_shifts using mean_shift_distance from all other orders.
    # In the first loop we obtain the missing shift_index and
    # the second loop moves based on the 1st order shift such that the 1st order is zero.
    # TODO: Is double looping over the order_index_shift efficient? (Prio Low)
    # seems a bit arbitrary, but provides plausible results.
    # Maybe there is a more efficient way.
    for on,shift in order_index_shift_dict.items():
        if shift is None:
            order_index_shift_dict[on] = order_index_shift_dict[on+1] - mean_shift_distance
    o1 = order_index_shift_dict[1]
    # Make that the 1st order has zero shift:
    for on,shift in order_index_shift_dict.items():
        order_index_shift_dict[on] = order_index_shift_dict[on] - o1

    print('shift dict:',order_index_shift_dict)
    # Compute left padding for all orders
    # We need to compute in reverse to have the longest order length available
    left_pad_dict = {}
    right_pad_dict = {}
    correction = params['manual_params']['aline_pad_correction_'+process_type]  # integer only
    for on in range(1,num_orders+1):
        correlation_index = order_index_shift_dict[on]
        left_pad_dict[on] = correlation_index + rescale_dict[on] + on * correction
        assert np.all(np.array(list(left_pad_dict.values()))>0), \
            '{}\nleft_pad_dict has negative values. Change aline_pad_correction_'.format(left_pad_dict, process_type)
    for on in range(1,num_orders):
        right_pad_dict[on] = (left_pad_dict[num_orders] + rescale_dict[num_orders]) - (left_pad_dict[on] + rescale_dict[on])
    right_pad_dict[num_orders] = 0

    # Plotting for analysing the computation of the pad values and goodness of overlaps.
    # Traverse all A-lines (outer loop)
    # Traverse all orders (inner loop)
    button_press.close = False
    # global_min = np.min(aline_orders_envelope)
    # aline_orders_envelope += global_min
    for al_nr,al in enumerate(aline_orders_envelope[150::-1]):
        for i,ol in enumerate(al):
            on = i+1
            pad = left_pad_dict[on]
            if revert_order:
                ol = ol[::-1]
            smpls = rescale_dict[on]
            vertical_offset = on * 0
            ol = 20*np.log10(abs(sg.resample(x=abs(ol), num=smpls))) + vertical_offset
            # make padding visible "pad_content=vertical_offset"
            # pad_content = vertical_offset
            pad_content = np.nan
            ol = np.pad(array=ol,pad_width=pad,mode='constant',constant_values=pad_content)[:-pad]
            ol = np.concatenate((ol, np.ones(right_pad_dict[on])*pad_content))

            # print('on:{:2d}, pad:{:5d}, len_new:{:5d}, rpad:{:5d}'.format(on, pad, len(ol), right_pad_dict[on]))

            # print into lines to allow zoom during plotting.
            if len(axs[1].get_lines()) < (on):
                axs[1].plot( ol )
            else:
                axs[1].get_lines()[on-1].set_ydata(ol)
            # mp.Axes.set_autoscale_on()
            # mp.Line2D.d
            # axs[1].plot( ol )
            # axs[1].semilogy( ol )
            # axs[1].set_ylim((1,15))
        # axs[1].set_autoscale_on(False)
        if debug_dict.get('show_all_orders') and debug_dict.get('do_stop2'):
            plt.draw()
            plt.waitforbuttonpress()
            if button_press.close: break
        elif debug_dict.get('show_all_orders'):
            plt.pause(0.1)
            if button_press.close: break

    if debug_dict.get('show_brief'):
        plt.pause(0.1)
        plt.draw()

    # Compute the padding length for the new arrays to be able to merge adjacent orders.
    # Please note that this does not account for additional array length of the lower orders without overlap.
    # Because we need to account for the additional space for the signal gaps.
    # The order displacement index computed above is the distance the smaller array as to move.
    # See ascii graphic below:
    # |------^---|     o_n
    # |---^---------|  o_(n+1)
    # The longer order must be padded to align the peaks
    # |------^---|     o_n
    #    |---^---------|  o_(n+1)
    #    |------>|     correlation index
    # The correlation index is the position where to move the end point of the shorter order.
    # To obtain the padding size we compute
    # pad_size = len(o_(n) - correlation_index
    # The first order does not need to be padded.


def plot_all_correlation_in_order_index_shift(num_orders, correlation_dict, order_index_shift_dict):
    """
    ...
    :return:
    """
    # plot all correlations and peaks
    # Please not that we ignore the order 1 as it has no correlation values (range(num_orders, 1, -1))!
    # We plot starting with the highest orders as those will have correlation values (reversed range).
    assert np.any([idx is None for idx in order_index_shift_dict.values()]), \
    'Parameter order_index_shift seems to be corrected.\n ' + \
    'You seem to call this plotting outside of \'compute_order_index_shift\' which is not intented.'

    fig, ax = subplots(nrows=1,ncols=1)
    plt.connect('key_press_event', button_press)
    button_press.fig = fig
    ax.set_title('Correlation graphs and detected peaks\n(Press button to proceed)')
    none_number_of_elements = 0
    for on in range(num_orders,1,-1):
        corr_graph = correlation_dict[on]
        shift_point = order_index_shift_dict[on]
        ax.plot(corr_graph)
        # negative values are gaps; but here we ignore them for the moment as we don't have correlation data.
        # so where no peak is we can't plot it.
        if shift_point is not None:
            ax.plot(shift_point,corr_graph[shift_point],'rx')
        else:
            none_number_of_elements += 1
            print('Order {} is ignored. No order_index_shift available.'.format(on))
    print('Order 1 is always ignored.')
    plt.draw()
    plt.waitforbuttonpress()


def plot_each_order_in_assemble(aline_orders_hilbert, rescale_dict, left_pad_dict, right_pad_dict, debug_dict):
    fig, ax = subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.connect('key_press_event', button_press)
    button_press.fig = fig
    ax.set_title('Visual placement of orders.')

    # Plotting for analysing the computation of the pad values and goodness of overlaps.
    # Traverse all A-lines (outer loop)
    # Traverse all orders (inner loop)
    for al_nr,al in enumerate(aline_orders_hilbert[150::2]):
        for i,ol in enumerate(al):
            on = i+1
            pad = left_pad_dict[on]
            smpls = rescale_dict[on]
            vertical_offset = on * 0
            ol = np.log10(abs(sg.resample(x=abs(ol), num=smpls))) + vertical_offset
            # make padding visible "pad_content=vertical_offset"
            # pad_content = vertical_offset
            pad_content = np.nan
            ol = np.pad(array=ol,pad_width=pad,mode='constant',constant_values=pad_content)[:-pad]
            ol = np.concatenate((ol, np.ones(right_pad_dict[on])*pad_content))

            # print('on:{:2d}, pad:{:5d}, len_new:{:5d}, rpad:{:5d}'.format(on, pad, len(ol), right_pad_dict[on]))

            # print into lines to allow zoom during plotting.
            if len(ax.get_lines()) < (on):
                ax.plot( ol )
            else:
                # ax.get_lines()[on-1].set_ydata(ol)
                ax.plot( ol )
            # mp.Axes.set_autoscale_on()
            # mp.Line2D.d
            # axs[1].plot( ol )
            # axs[1].semilogy( ol )
            # axs[1].set_ylim((1,15))

            # axs[1].set_autoscale_on(False)
            plt.draw()
            plt.waitforbuttonpress()
            # This is currently highly experimental view showing each three orders unscaled.
            if len(ax.get_lines()) > 2:
                ax.get_lines()[0].remove()
            if hasattr(button_press, 'close') and button_press.close is True:
                break


def plot_merge_order_in_assemble(params,aline_orders_hilbert, rescale_dict, left_pad_dict, right_pad_dict, is_reversed=False):
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
    fig, ax = subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.connect('key_press_event', button_press)
    button_press.fig = fig
    ax.set_title('Visual placement of orders.')
    bframe = None
    filterwarnings(action='ignore', category=RuntimeWarning)

    revert_order = is_reversed
    if revert_order: warn('Reverting orders is active')
    print('Start assembling B-frame:')
    print('         ',end='') # print some progress number
    for al_nr,al in enumerate(aline_orders_hilbert[0::1]):
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

            # aline = np.nansum((aline, ol), axis=0)
            aline = np.nanmean((aline, ol), axis=0)

        if bframe is None:
            bframe = aline
        else:
            bframe = np.vstack((bframe, aline))
        # print(np.shape(bframe))
        # print(aline)
        # print into lines to allow zoom during plotting.
        if len(ax.get_lines()) < 1:
            ax.semilogy( aline )
            ax.set_ylim((1e-2,25000))
        else:
            ax.get_lines()[0].set_ydata(aline)

        # ax.autoscale()
        # plt.draw()
        # plt.waitforbuttonpress()
        # ax.cla()

        if hasattr(button_press,'close') and button_press.close is True:
            break
        # axs[1].set_autoscale_on(False)
        # plt.draw()
        # plt.waitforbuttonpress()
        # ax.cla()

    print('') # print final for progress
    print('Assembling B-frame done.')
    bframe[np.where(np.isnan(bframe))] = np.nanmedian(bframe)
    # bframe = np.take(1e-12,nan_idx)# set to really small value where no data are
    # bframe[np.nonzero(bframe == np.nan)] = 1e-12 # set to really small value where no data are
    bframe = bframe.T
    figure()
    plt.imshow(20*np.log10(bframe),aspect=bframe.shape[1]/bframe.shape[0],cmap='CMRmap',vmin=10,interpolation='none')
    plt.colorbar()
    plt.show()


def guess_str_from_path(path, stop_pattern='2020'):
    """
    Construct a title string based on the path trying to be smart about what to keep and remove
    if it is too long.
    :param path:
    :param stop_pattern: A string pattern to cut off meaningless path name
    :return:
    """
    # Guess some title from path (adapt find where roughly to break)
    for n, pi in enumerate([ps for ps in path.split(os.path.sep)][::-1]):
        if pi.lower().find(stop_pattern) > -1: break
    tstr = path.split(os.path.sep)[-n - 2::]
    if 'data.bin' in tstr[-1]: del tstr[-1]  # remove meaningless fname
    tstr = ' '.join(tstr)
    return tstr

def break_title_str(tstr, new_line_at=30, reduce_at=60):
    """
    Try to find reasonable line breaks in title string automatically for optimal plotting.
    :param new_line_at:
    :param reduce_at:
    """
    sep_idx = tstr[len(tstr)//2:].rfind(' ')
    # if len(tstr) > new_line_at and len(tstr) <= reduce_at:
    #     p1 = tstr[:sep_idx]
    #     p2 = tstr[sep_idx:]
    #     tstr = p1 + '\n' + p2
    # elif len(tstr) > reduce_at:
    if len(tstr) > reduce_at*2:
        p1 = tstr[:reduce_at//2]
        p2 = tstr[reduce_at//2:]
        tstr_sp = p1 + '...\n...' + p2
    elif len(tstr) > reduce_at:
        p1 = tstr[:len(tstr)//2]
        p2 = tstr[len(tstr)//2:]
        tstr_sp = p1 + '\n' + p2
    else:
        tstr_sp = tstr

    return tstr_sp


def plot_image_reconstructed(params, bframe_fw, bframe_rv):

    # rolling and resampling for aligning must be last step
    bframe_rv = mf.sg.resample(x=bframe_rv, num=int(bframe_fw.shape[0]*params['manual_params']['fw_resample']), window='nuttall')
    assert np.min(bframe_rv) > 0, 'Parameter fw_resample generates negative values. Either change it or try another window.'
    bframe_rv = mf.np.roll(a=bframe_rv,shift=int(params['manual_params']['fw_roll']),axis=0)

    # Create the target matrix based on the largest shape
    if bframe_fw.shape[0] >= bframe_rv.shape[0]:
        bframe = np.empty((bframe_fw.shape[0], bframe_fw.shape[1] + bframe_rv.shape[1]))
    elif bframe_rv.shape[0] > bframe_fw.shape[0]:
        bframe = np.empty((bframe_rv.shape[0], bframe_fw.shape[1] + bframe_rv.shape[1]))
    else:
        raise ('Something else went wrong. Can\'t find final common bframe width.')

    # plt.imshow(bframe_fw,aspect=1/2,vmin=0,vmax=15000); plt.show()

    median_noise_fw = np.nanmedian(bframe_fw)
    median_noise_rv = np.nanmedian(bframe_rv)

    # bframe_fw[np.nonzero(bframe_fw<=0)]=np.nanmin(bframe_fw) #params['manual_params']['nan_value']
    # bframe_rv[np.nonzero(bframe_rv<=0)]=np.nanmin(bframe_rv) #params['manual_params']['nan_value']
    # can also use np.nan_to_num(arr, nan= nan_value)

    bframe[:bframe_fw.shape[0], 0::2] = bframe_fw
    bframe[:bframe_rv.shape[0], 1::2] = bframe_rv


    # mf.np.save('bframe',bframe)
    # bframe = mf.np.load('bframe.npy')
    time_stamp = params['auto_params']['time_stamp']
    title_str = break_title_str(params['auto_params']['title_str'])

    def default_fig_config():
        fig = plt.gcf()
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, cbar_mode='single')
        imax.axes.set_title(  'fw '+time_stamp+'\n'+title_str)
        cax = grid.cbar_axes[0]
        fig.colorbar(mappable=imax, cax=cax)
        plt.tight_layout()

    figure(num='Roll-off\n'+time_stamp)
    plt.title('Roll-off '+time_stamp+'\n'+title_str)
    median_noise = np.nanmean((median_noise_fw,median_noise_rv))
    median_noise_dB = 20*np.log10(median_noise)
    print('median noise: {} dB'.format(median_noise_dB))
    roll_off = 20*np.log10(bframe[:,::5].max(axis=0))
    #roll_off[np.nonzero(roll_off<60)] = np.nan
    plt.plot(roll_off)
    plt.axhline(y=median_noise_dB)
    plt.tight_layout()


    figure(num='merged PSF\n'+time_stamp)
    plt.title('PSF merged '+time_stamp+'\n'+title_str)
    plt.plot(20*np.log10(bframe[:,::15]),lw=0.75)
    #plt.ylim((0,100))
    plt.tight_layout()

    # fig = figure(num='fw '+time_stamp)
    # # plt.gcf().canvas.manager.window.move(100,100)
    # # https://github.com/matplotlib/matplotlib/issues/4282
    # # for colorbar and aspect change the link describes the currently only way. But check later matplotlib versions.
    # grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, cbar_mode='single')
    # imax = grid[0].imshow(20 * np.log10(bframe_fw),
    #            aspect=bframe_fw.shape[1] / bframe_fw.shape[0],
    #            cmap='Greys_r',
    #            # cmap='CMRmap',
    #            vmin=0, vmax=90,
    #            interpolation='none')
    #
    # imax.axes.set_title(  'fw '+time_stamp+'\n'+title_str)
    # cax = grid.cbar_axes[0]
    # fig.colorbar(mappable=imax, cax=cax)
    # plt.tight_layout()
    # fm.set_window_title('can set num afte')

    # fig = figure(num='rv '+time_stamp)
    # # plt.gcf().canvas.manager.window.wm_geometry("+650+0") # for TkAgg
    # # plt.gcf().canvas.manager.window.move(600,100)
    # grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, cbar_mode='single')
    # imax = grid[0].imshow(20 * np.log10(bframe_rv),
    #            aspect=bframe_rv.shape[1] / bframe_rv.shape[0],
    #            cmap='Greys_r',
    #            # cmap='CMRmap',
    #            vmin=0, vmax=90,
    #            interpolation='none')
    # imax.axes.set_title(  'rv '+time_stamp+'\n'+title_str)
    # cax = grid.cbar_axes[0]
    # fig.colorbar(mappable=imax, cax=cax)
    # plt.tight_layout()

    fig = figure(num='merged ' + time_stamp)
    # plt.gcf().canvas.manager.window.wm_geometry("+300+300")
    # plt.gcf().canvas.manager.window.move(300,300)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.1, cbar_mode='single')
    imax = grid[0].imshow(20 * np.log10(bframe),
               # aspect = 2,
               # aspect=bframe.shape[1] / bframe.shape[0],
               # cmap='Greys_r',
               cmap='Greys_r',
               #vmin=30, vmax=90,
               interpolation='none', # really none
               # interpolation=None, # some default
               # interpolation='antialiased', # removes morrie
               extent = (0,params['manual_params']['lateral_scan_width'],params['manual_params']['scan_range']*params['manual_params']['filter']['use_num_orders'],0)
               )
    imax.axes.set_title('merged ' + time_stamp+'\n'+title_str)
    imax.axes.set_xlabel('(μm)')
    imax.axes.set_ylabel('(μm)')
    cax = grid.cbar_axes[0]
    fig.colorbar(mappable=imax, cax=cax)
    plt.tight_layout()
    plt.show()
