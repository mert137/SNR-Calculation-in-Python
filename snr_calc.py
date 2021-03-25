from __future__ import division
import xlrd
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

filetype = "txt"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__filename__ = 'sample.txt'
L = 2048
f_s = 250e6
peakToPeakVoltageLevel = 2
ADC_resolution_bits = 14

if filetype == "xls":
###########################   Read xls      ##############################################
    wb = xlrd.open_workbook(os.path.join(__location__, __filename__))
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    partialADCRawData = np.ones(L)

    for i in range(sheet.nrows):
        partialADCRawData[i] = sheet.cell_value(i, 0)
else:
###########################   Read txt      ##############################################
	f = open(os.path.join(__location__, __filename__))
    fl = f.readlines()

    partialADCRawData = np.ones(L)
    i = 0

    for x in fl:
        partialADCRawData[i] = x
        i = i + 1





###########################   Time Domain   ##############################################

T = 1 / f_s  # Sampling period
t = np.arange(L) * T  # Time vector
fft_process_gain = 10*np.log10(L/2)

plotData = (partialADCRawData * (peakToPeakVoltageLevel / (2 ** ADC_resolution_bits)))  # - (peakToPeakVoltageLevel/2)





###########################   Remove DC    ################################################

min_data = np.min(plotData)
max_data = np.max(plotData)
center_data = (min_data + max_data) / 2
plotData = plotData - np.mean(plotData)


f1 = plt.figure(1)
plt.plot(t, plotData)
plt.xlabel('Time [s]')
plt.ylabel('Signal amplitude')
f1.show()





###########################   FFT linear   ################################################

fft_linear = fftpack.fftshift(np.abs(fftpack.fft(plotData)) / (L/2))
fft_linear_one_side = fft_linear[int(L/2):L]
fft_linear_one_side[0] = fft_linear_one_side[0] / 2
freqs = np.linspace(0, f_s / 2, int(L/2)) / 1e6

fft_linear_one_side[0] = 1e-7  # DC component is removed above. The first term which is DC component is assigned small value so that it is not equal to 0.

V_p = np.max(fft_linear_one_side)

###########################   FFT dB  ####################################################

fft_dB = 20 * np.log10(fft_linear_one_side / (peakToPeakVoltageLevel/2))     # dBFS

f3 = plt.figure(3)
plt.plot(freqs, fft_dB)
plt.xlabel('Frequency in Hertz [MHz]')
plt.ylabel('Frequency Domain (Spectrum) Magnitude(dB)')
plt.xlim(0, max(freqs))
plt.ylim(-150, 15)
f3.show()





###########################___Noise_Floor   ##############################################

theset = frozenset(fft_dB)
theset = sorted(theset, reverse=True)
limit_dB = theset[5]

#limit_dB = -92.3

noise_sum_dB = 0
k = 0
for i in range(len(fft_dB)):
    if fft_dB[i] < limit_dB:
        noise_sum_dB = noise_sum_dB + fft_dB[i]
        k = k + 1

noise_floor = noise_sum_dB / k
nsd = noise_floor + fft_process_gain - 10 * np.log10(f_s/2)

##########################################################################################

fft_dB_noise_floor = np.copy(fft_dB)
for i in range(len(fft_dB_noise_floor)):
    if fft_dB_noise_floor[i] > limit_dB:
        fft_dB_noise_floor[i] = noise_floor

f5 = plt.figure(5)
plt.plot(freqs, fft_dB_noise_floor)
plt.xlabel('Frequency in Hertz [MHz]')
plt.ylabel('Frequency Domain (Spectrum) Magnitude(dB)')
plt.xlim(0, max(freqs))
plt.ylim(-150, 15)
f5.show()

# np.delete(fft_linear_one_side, np.argmax(fft_linear_one_side))    # delete dc component
x_sum = max(fft_linear_one_side) ** 2                    # square of original signal
n_sum = 0

for i in range(len(fft_dB_noise_floor)):
    n_sum = n_sum + (10 ** (fft_dB_noise_floor[i] / 20)) ** 2

V_pp = V_p * 2
V_rms = V_p / np.sqrt(2)
R = 27
dBm = 10 * np.log10(V_rms**2 / R) + 30

##########################################################################################

SNR_rms = 10 * np.log10(x_sum / n_sum)
SNR_fft = max(fft_dB) - noise_floor - fft_process_gain
a = 0