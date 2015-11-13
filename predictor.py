import sklearn
from sklearn import svm

def train_and_test(training_vectors, training_labels, test_vectors, test_labels):
    classifier = svm.SVC(C=50) # (C=50, gamma=1)
    #print 'training...'
    #print training_labels
    classifier.fit(training_vectors, training_labels)

    correct = 0
    #print 'testing...'
    #for index, vector in enumerate(test_vectors):
    #    prediction = classifier.predict(vector)
    #    actual = test_labels[index]
    #    print 'prediction: {prediction}, actual: {actual}'.format(
    #        prediction=prediction,
    #        actual=actual
    #    )
    #    if prediction == actual:
    #        correct += 1

    #accuracy = 100 * (correct / float(len(test_vectors)))
    #print 'accuracy: {accuracy}%'.format(accuracy=accuracy)

    score = classifier.score(test_vectors, test_labels)
    print 100 * score, '%'
    num_right = int(score * len(test_vectors))
    num_wrong = len(test_vectors) - num_right
    return num_right, num_wrong
