import sklearn
from sklearn import svm

def train_and_test(training_vectors, training_labels, test_vectors, test_labels):
    classifier = svm.SVC(C=50) # (C=50, gamma=1)

    classifier.fit(training_vectors, training_labels)

    score = classifier.score(test_vectors, test_labels)
    # print classifier.decision_function(test_vectors)
    print 100 * score, '%'
    num_right = int(score * len(test_vectors))
    num_wrong = len(test_vectors) - num_right
    return num_right, num_wrong
