import sys
from feature_extractor import *
from xml.dom.minidom import parse
import pycrfsuite

datadir = sys.argv[1]
outfile = sys.argv[2]

X_train = []
y_train = []

i = 0

for f in listdir(datadir):
    # parse XML file , obtaining a DOM tree
    tree = parse(datadir + "/" + f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value # get sentence id
        stext = s.attributes["text"].value # get sentence text
        # load ground truth entities .
        gold = []
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities , we only get the first span
            offset = e.attributes["charOffset"].value
            (start, end) = offset.split(";")[0].split("-")
            gold.append((int(start), int(end), e.attributes["type"].value))
        # tokenize text
        tokens = tokenize(stext)
        X_train[i] = extract_features(tokens)
        y_train[i] = [get_tag(token, gold) for token in tokens]



trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('trainer.crfsuite')

# TODO make predictions

testdir = './data/devel/'

tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')

for f in listdir(testdir):
    test_tree = parse(testdir + "/" + f)
    test_sentences = tree.getElementsByTagName("sentence")
    test_texts = []

    for s in sentences:
        sid = s.attributes["id"].value # get sentence id
        stext = s.attributes["text"].value # get sentence text
        test_texts.append(stext)
        # load ground truth entities .
        gold = []
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities , we only get the first span
            offset = e.attributes["charOffset"].value
            (start, end) = offset.split(";")[0].split("-")
            gold.append((int(start), int(end), e.attributes["type"].value))
        # tokenize text
        tokens = tokenize(stext)
        print("Predicted:", ' '.join(tagger.tag(extract_features(tokens))))
        print("Correct:  ", ' '.join([get_tag(token, gold) for token in tokens]))