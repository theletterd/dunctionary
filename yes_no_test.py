import os

from predictor import train_and_test
from vectoriser import get_normalised_vector


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
def get_labelled_yn_data_for_person(person_id):
    # british is 1:37
    # print 'getting labels for', person_id
    if person_id in CACHED_DATA:
        yes_instances = CACHED_DATA[person_id]['yes']
        no_instances = CACHED_DATA[person_id]['no']
    else:
        yes_instances = get_instances_from_dir(
            '/home/duncan/Dropbox/Duncan/Dunctionary/samples/p{person_id}/yes/'.format(person_id=person_id)
        )

        no_instances = get_instances_from_dir(
            '/home/duncan/Dropbox/Duncan/Dunctionary/samples/p{person_id}/no/'.format(person_id=person_id)
        )
        CACHED_DATA[person_id] = {'yes': yes_instances, 'no': no_instances}

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

    return train_and_test(training_vectors, training_labels, test_vectors, test_labels)

def run_yn_test():
    total_right, total_wrong = 0, 0
    for i in xrange(1, 38):
        try:
            right, wrong = test_person(i)
            total_right += right
            total_wrong += wrong
        except Exception, e:
            raise e
            pass

    print total_right, total_wrong
    print 100 * (total_right / float(total_right + total_wrong))

if __name__ == '__main__':
    run_yn_test()
