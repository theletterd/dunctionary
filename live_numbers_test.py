from sound_recorder import get_raw_wav_data
from number_test import get_labelled_number_data_for_person
from predictor import train_with_data
from vectoriser import get_normalised_vector

#1 - train classifier
training_vectors, training_labels = [], []
for person_index in xrange(1, 38):
    vectors, labels = get_labelled_number_data_for_person(person_index)
    training_vectors += vectors
    training_labels += labels

print 'training...'
classifier = train_with_data(training_vectors, training_labels)

while True:
    #2 - get raw data
    sample_rate, data = get_raw_wav_data()

    #3 vectorise data
    vector = get_normalised_vector(sample_rate, data)

    #4 test data
    prediction = classifier.predict(vector)[0]
    print '\n\n\n\n\n'
    print prediction

    print '\n\n\n\n\n'
