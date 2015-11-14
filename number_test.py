import os

from predictor import train_and_test
from predictor import train_with_data
from vectoriser import get_normalised_vector
from logistic_regression import get_curve_params
from logistic_regression import probability

NUMBERS = [
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine'
]



def get_instances_from_dir(path):
    filenames = os.listdir(path)
    arrays = []
    for filename in filenames:
        if filename.endswith('.wav'):
            f_name = path + filename
            try:
                vect = get_normalised_vector(filename=f_name)
            except Exception, e:
                continue

            arrays.append(vect)

    return arrays

CACHED_DATA = {}
def get_labelled_number_data_for_person(person_id):
    vectors, labels = [], []

    if person_id in CACHED_DATA:
        for label, number in enumerate(NUMBERS):
            instances = CACHED_DATA[person_id][number]

            vectors += instances
            labels += [label] * len(instances)
    else:
        CACHED_DATA[person_id] = {}
        for label, number in enumerate(NUMBERS):
            instances = get_instances_from_dir(
                '/home/duncan/Dropbox/Duncan/Dunctionary/samples/p{person_id}/{number}/'.format(
                    person_id=person_id,
                    number=number
                )
            )
            CACHED_DATA[person_id][number] = instances

            vectors += instances
            labels += [label] * len(instances)

    return vectors, labels

def test_person(person_id):
    british_ids = set(xrange(1, 48))
    test_person = person_id
    british_ids.remove(test_person)
    print 'Test Person:', test_person

    test_vectors, test_labels = get_labelled_number_data_for_person(test_person)
    if not test_vectors:
        print 'No Samples for person ', person_id, ' - skipping'
        return 0, 0
    training_vectors, training_labels = [], []

    for person_id in british_ids:
        vectors, labels = get_labelled_number_data_for_person(person_id)

        training_vectors += vectors
        training_labels += labels
    return train_and_test(training_vectors, training_labels, test_vectors, test_labels)


def run_numbers_test():
    total_right, total_wrong = 0, 0
    for i in xrange(1, 48):
        try:
            right, wrong = test_person(i)
            total_right += right
            total_wrong += wrong
        except Exception, e:
            raise e
            pass

    print total_right, total_wrong
    print 100 * (total_right / float(total_right + total_wrong))

class ProbabilisticSVM(object):

    def __init__(self, classifier, curve_data):
        self.classifier = classifier
        self.curve_data = curve_data

    def get_probability(self, vector):
        [score] = self.classifier.decision_function([vector])
        return probability(self.curve_data, score)

def train_one_vs_all_models(person_id):
    prob_classifiers = []

    #score_person = 4
    test_person = person_id

    person_ids = set(xrange(1, 48))
    #person_ids.remove(score_person)
    person_ids.remove(test_person)

    for number in xrange(10):

        # create the training data
        training_vectors, training_labels = [], []
        for person_id in person_ids:
            vectors, labels = get_labelled_number_data_for_person(person_id)
            labels = [int(label == number) for label in labels]
            training_vectors += vectors
            training_labels += labels

        # get the scoring data
        #score_vectors, score_labels = get_labelled_number_data_for_person(score_person)
        #score_labels = [int(label == number) for label in score_labels]

        classifier = train_with_data(training_vectors, training_labels)
        #scores = classifier.decision_function(score_vectors)
        scores = classifier.decision_function(training_vectors)

        #yes_scores = [scores[index] for index in xrange(len(scores)) if score_labels[index] == 1]
        #no_scores = [scores[index] for index in xrange(len(scores)) if score_labels[index] == 0]

        yes_scores = [scores[index] for index in xrange(len(scores)) if training_labels[index] == 1]
        no_scores = [scores[index] for index in xrange(len(scores)) if training_labels[index] == 0]

        curve_data = get_curve_params(yes_scores, no_scores)
        prob_classifiers.append(ProbabilisticSVM(classifier, curve_data))

    # now let's hit up the classifiers with the test data
    test_vectors, test_labels = get_labelled_number_data_for_person(test_person)

    hits = 0

    for vector, label in zip(test_vectors, test_labels):
        probabilities = []
        for index, classifier in enumerate(prob_classifiers):
            probabilities.append((classifier.get_probability(vector), index))
        probabilities.sort(reverse=True)

        hit = probabilities[0][1] == label
        if hit:
            hits += 1

    print 100 * (hits / float(len(test_labels))), '%'
    return hits, len(test_labels)
    #test_scores = classifier.decision_function(test_vectors)
    #probabilities = [probability(curve_data, score) for score in test_scores]
    #probabilities_and_labels = zip(probabilities, test_labels)
    #print sorted(probabilities_and_labels, key=lambda derp: derp[1])
    #import ipdb; ipdb.set_trace()
    # train a model.
    # get scores for curve for p4
    # make predictions for p8

def full_gamut():
    total_hits, total_length = 0, 0

    for person_id in xrange(1, 48):
        hits, length = train_one_vs_all_models(person_id)
        total_hits += hits
        total_length += length

    print 100 * (total_hits / float(total_length))

if __name__ == '__main__':
    run_numbers_test()
    #train_one_vs_all_models()
    #full_gamut()
