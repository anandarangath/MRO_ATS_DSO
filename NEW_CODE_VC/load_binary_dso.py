import numpy as np
import matplotlib.pyplot as plt

# # fname = r"C:\Users\anand\National University of Ireland, Galway\Group_TOMI Lab - MRO_DUAL_DET\Test_09_03_2021\ATS_DSO\11_03_2021\buffer_35032\1\2021.03.11_15.30.45_1.C.bin"
# fname = r"C:\Users\anand\National University of Ireland, Galway\Group_TOMI Lab - MRO_DUAL_DET\ATS_DSO\12_03_2021\1\2021.03.12_15.37.18_1.C.bin"
#
# buffer_length_orig = 35032
# data_full = np.fromfile(open(fname,'rb'),dtype=('f8',buffer_length_orig))
# print("The shape of data is", np.shape(data_full))
#
# DC_removed = data_full - np.mean(data_full)
# DC_removed = DC_removed[:, 200:]
#
# for A_line_orig in DC_removed:
#     A_line = A_line_orig
#     buffer_length = len(A_line)
#     fw = A_line[:buffer_length//2]
#     rv = A_line[buffer_length//2 :]
#     plt.subplot(121)
#     plt.plot(fw)
#     plt.subplot(122)
#     plt.plot(rv)
#     plt.waitforbuttonpress()
#     plt.clf()

phase_fw = np.load('phase_fw.npy')
phase_rv = np.load('phase_rv.npy')

print('min and max in fw', np.min(phase_fw), np.max(phase_fw))
print('min and max in rv', np.min(phase_rv), np.max(phase_rv))