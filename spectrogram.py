import os
from scipy.io.wavfile import read
from scipy.signal import periodogram
from scipy.signal import get_window
from matplotlib.mlab import specgram
from sklearn.preprocessing import normalize
from scipy.misc import imresize

import matplotlib.pyplot as plt

import math
import numpy


MAX_FREQ = 4000
OUTPUT_ROWS = 20
OUTPUT_COLS = 50

def show_plot(spectrum):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(spectrum, interpolation='nearest', cmap='Greys')
    plt.show()

def convert_frequency_to_mel(freq):
    return (1127 * math.log(1 + freq / 700.0))

def convert_mel_to_frequency(mel):
    return 700 * (math.exp(mel/1127) - 1)

def get_power_spectrum_of_wave(filename):
    sample_rate, data = read(filename)

    noverlap = 96
    nfft = 128

    spectrum, frequencies, times = specgram(
        data,
        Fs=sample_rate,
        noverlap=noverlap,
        NFFT=nfft,
        scale_by_freq=True,
    )

    return spectrum

def scale_power_spectrum_to_mel_scale(spectrum):
    num_rows = len(spectrum)
    max_mel = math.floor(convert_frequency_to_mel(MAX_FREQ))
    mels_per_bucket = math.floor(max_mel / OUTPUT_ROWS)

    # repeat each vector
    repeat_factor = math.ceil(MAX_FREQ / num_rows)
    repeated_matrix = numpy.repeat(spectrum, repeat_factor, axis=0)

    bucket_start_end_indices = numpy.floor(
        numpy.array(
            map(
                convert_mel_to_frequency,
                numpy.array(
                    xrange(OUTPUT_ROWS + 1)
                ) * mels_per_bucket
            )
        )
    ).astype(int)
    # get start of each mel bucket
    row_coordinates = [
        (bucket_start_end_indices[i], bucket_start_end_indices[i + 1])
        for i in xrange(len(bucket_start_end_indices) - 1)
    ]

    scaled_matrix = numpy.array([
        sum(repeated_matrix[coord[0]:coord[1], :]) for coord in row_coordinates
    ])

    return scaled_matrix


def trim_start_end_silence(spectrum):
    normalized = normalize(spectrum)
    mean = numpy.mean(normalized)
    relevant_indices = numpy.where(numpy.mean(normalized, axis=0) > 0.1) #
    first_index = relevant_indices[0][0]
    last_index = relevant_indices[0][-1]

    trimmed_spectrum = normalized[:, first_index:last_index + 1]
    return trimmed_spectrum

def standardise_to_dimensions(trimmed):
    resized = imresize(trimmed, (OUTPUT_ROWS, OUTPUT_COLS))
    # could re-normalize here also.
    #show_plot(resized)
    return normalize(resized.astype(float))

def get_normalised_vector(fname):
    spectrum = get_power_spectrum_of_wave(fname)
    scaled_spectrum = scale_power_spectrum_to_mel_scale(spectrum)
    normalized = normalize(scaled_spectrum)
    trimmed = trim_start_end_silence(normalized)
    standardised = standardise_to_dimensions(trimmed)
    return standardised.flatten()

def instances(path):
    filenames = os.listdir(path)
    arrays = []
    for filename in filenames:
        if filename.endswith('.wav'):
            f_name = path + filename
            try:
                vect = get_normalised_vector(f_name)
            except Exception, e:
                continue

            arrays.append(vect)

    return arrays


def get_labelled_yn_data_for_person(person_id):
    # british is 1:37
    # print 'getting labels for', person_id
    yes_instances = instances(
        '/home/duncan/Dropbox/Duncan/Dunctionary/samples/p{person_id}/yes/'.format(person_id=person_id)
    )

    no_instances = instances(
        '/home/duncan/Dropbox/Duncan/Dunctionary/samples/p{person_id}/no/'.format(person_id=person_id)
    )

    vectors = yes_instances + no_instances
    labels = ([1] * len(yes_instances)) + ([0] * len(no_instances))
    return vectors, labels

def test_person(person_id):
    british_ids = set(xrange(1, 38))
    test_person = person_id
    british_ids.remove(test_person)
    print 'Test Person:', test_person

    test_vectors, test_labels = get_labelled_yn_data_for_person(test_person)
    if not test_vectors:
        print 'No Samples for person ', person_id, ' - skipping'
        return 0, 0
    training_vectors, training_labels = [], []

    for person_id in british_ids:
        vectors, labels = get_labelled_yn_data_for_person(person_id)

        training_vectors += vectors
        training_labels += labels

    from predictor import train_and_test

    return train_and_test(training_vectors, training_labels, test_vectors, test_labels)

total_right, total_wrong = 0, 0
for i in xrange(1, 38):
    try:
        right, wrong = test_person(i)
        total_right += right
        total_wrong += wrong
    except Exception, e:
        pass

print total_right, total_wrong
print 100 * (total_right / float(total_right + total_wrong))
