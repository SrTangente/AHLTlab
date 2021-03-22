import sys
import pycrfsuite


if __name__ == "__main__":

    try:
        model = sys.argv[1]
        features = sys.argv[2]
    except:
        model = 'model.crfsuite'
        features = 'train.feat'

    X_train = []
    y_train = []

    with open(features, "r") as file:
        line = file.readline()

        while line:
            sentence_features = []
            sentece_tags = []
            while line != '\n':
                split = line.split('\t')
                feat = split[5:]
                tag = split[4]
                sentence_features.append(feat)
                sentece_tags.append(tag)
                line = file.readline()

            X_train.append(sentence_features)
            y_train.append(sentece_tags)
            line = file.readline()

    algorithm = "l2sgd"
    params = {
        #'c1': 1.0,   # coefficient for L1 penalty
        #'c2': 1e-4,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    }
    trainer = pycrfsuite.Trainer(algorithm=algorithm, verbose=True)
    #print(trainer.params())
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params(params)

    trainer.train(model)
    print(trainer.logparser.iterations[-1])
