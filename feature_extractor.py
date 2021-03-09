from Baseline import *
from xml.dom.minidom import parse

datadir = './data/train/'
OUTFILENAME = './results.txt'


def get_tag(token, gold):
    for triplet in gold:
        if token[1] == triplet[0]:
            return 'B'+triplet[2]
        elif token[1] >= triplet[0] and token[2] <= triplet[1]:
            return 'I'+triplet[2]
    return 'O'


def extract_features(tokens):
    features = []
    for i in range(len(tokens)):
        #TODO extract features of each token
    return features


# process each file in directory
for f in listdir(datadir):
    # parse XML file , obtaining a DOM tree
    tree = parse(datadir + "/" + f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value # get sentence id
        stext = s.attributes[" text "].value # get sentence text
        # load ground truth entities .
        gold =[]
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities , we only get the first span
            offset = e.attributes["charOffset"].value
            (start, end) = offset.split(";")[0].split("-")
            gold.append((int(start) , int(end) , e.attributes["type"].value))
        # tokenize text
        tokens = tokenize(stext)
        # extract features for each word in the sentence
        features = extract_features(tokens)
        # print features in format suitable for the learner / classifier
        for i in range(0, len(tokens)):
            # see if the token is part of an entity , and which part (B/I)
            tag = get_tag(tokens[i], gold)
            print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')
        # blank line to separate sentences
        print()