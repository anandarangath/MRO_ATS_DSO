
from NEW_CODE_VC import mro_plotting as mp
from NEW_CODE_VC import mro_functions as mf
from NEW_CODE_VC import mro_functions_debug as mf
from NEW_CODE_VC.mro_manual_params import *
import shutil
import time
import os

time_stamp = time.strftime('%Y%m%d_%H%M%S',time.localtime())

params = mf.load_json_params( time_stamp )

params, data = mf.load_data(params, calib_fname, img_fname)

params = mf.align_mirror_data_and_save_phase(params, data )

params, fw_data, rv_data = mf.align_img_data_fw_rv(params, data)

params, linear_fw_data = mf.linearize_signal(params=params, data_aligned=fw_data, phase_type='fw')
params, linear_rv_data = mf.linearize_signal(params=params, data_aligned=rv_data, phase_type='rv')

params = mf.update_json_filter_frequencies(params, linear_fw_data, linear_rv_data)

params, linear_fw_data = mf.signal_apodization(params, linear_fw_data, process_type='fw')
params, linear_rv_data = mf.signal_apodization(params, linear_rv_data, process_type='rv')

params, linear_fw_data = mf.resample_signal(params, linear_fw_data, process_type='fw')
params, linear_rv_data = mf.resample_signal(params, linear_rv_data, process_type='rv')

params,aline_fw_orders = mf.filter_signal(params, linear_fw_data, process_type='fw')
params,aline_rv_orders = mf.filter_signal(params, linear_rv_data, process_type='rv')

params, aline_fw_orders_hlb = mf.hilbert_all_aline_orders(params, aline_fw_orders, process_type='fw')
params, aline_rv_orders_hlb = mf.hilbert_all_aline_orders(params, aline_rv_orders, process_type='rv')

# TODO: If image processing mode is used the orders can be reversed
# Can this be detected automatically?
if params['auto_params']['is_calibration']:
    reverse_bool = False
else:
    reverse_bool = False

params, bframe_fw = mf.assemble_orders_to_alines(params, aline_fw_orders_hlb, is_reversed=reverse_bool, process_type='fw')
params, bframe_rv = mf.assemble_orders_to_alines(params, aline_rv_orders_hlb, is_reversed=not (reverse_bool), process_type='rv')


mp.plot_image_reconstructed(params, bframe_fw, bframe_rv )

# todo: Investigate the use of FFT and ??? to evaluate the position of frequencies in a scattering sample.
# There seems to be strong indications that the refractive index affects the center frequencies.
# But at the moment it should be measured.
#
# todo: Filter setttings for each fw and rv
#
# todo: cut-off settings for each fw and rv

# todo: after background subtraction the intensity steps require equalization.
# The intensity steps should be possible to be extracted from the background profile.
# For that we may want to extend the analysis plot for order assembling to visualize the formation of intensity.
# But we could also assemple a peak intensity plot for all orders and see if we can measure this.

# todo: investigate equalization between orders
# If we can extract the intensity differences similar like median background then we could subtract the intensity
# levels not matching. Maybe like a localized median subtraction.

# todo: Can we correlate the forward and reverse? See comment
# For mirror data it seems possible to compare adjacent A-line shapes and use correlation to determine if a signal
# is present. Such a correlation can also amplify the signal - improving SNR.
# >>> we tested this and a full correlation of the whole segment or A-line does not improve SNR.
# >>> But see todo below.
# Does this make sense for scattering samples?
# >>> Not if correlating the whole segment. But using smaller segments or windows it still might present some advantage.
# If the signals are weak and close to noise correlation may create false positives.
# >>> Yes there is a statistical probability that we can find a model signal in any noise pattern.
# >>> However, this depends where the model search is performed.
# >>> Because, even though a pattern may be found it will not match adjacent A-lines and present a noise artefact again.
# >>> A model search is related to correlating a PSF signal with a whole segment.

# todo: Correlation using small segments/windows to reduce samples and increase SNR.
# This is a bit like cmOCT but with the difference that we not look for flow but for signals within the noise.
# The overlapping regions should have small windows or segments that match in shape while other do not.
# The caveat or balance that would need to be found is to by how much the down-sampling before filtering should be,
# which decimates the available samples.
# Whereas, such decimation using 1D correlation window-segments could help here to reduce noise before filtering,
# if the window-length is still smaller than the smallest wavelengths (highest frequencies).

# todo: Computation of physical image dimensions
# The mirror step width gives us some means to compute the aspect ratio of the pixel.
# In the same manner, the pixel width for each A-line must be given directly as it can not be deduced directly.
# The lateral scan width depends on the scanning angle of the galvo mirror vs. the distance to the sample, including
# the effect of the focusing lens.

# todo: Background subtraction of the intensity image (Evaluation, Differences, other)
# Background subtraction of an intensity image requires more sophisticated processing not to remove too much of
# the actualy image information.
# Any profile that can be extracted will subtract some pixel such that they become negative.
# Pixels with negative values can not be disblayed reliably and actually lost.
# To avoid pushing pixels into negative values scaling for each background pixel would need to be computed.
# But this is not special anymore and we can use available other algorithms.
# The actual idea was that we have more a priory knowledge about the background due to access of the signal, but this
# seems not entirely true.