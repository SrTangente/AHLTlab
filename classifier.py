import sys
import pycrfsuite


def output_entities(sid, tokens, tags):
    last_tag = "O"
    i = 0
    while i < len(tags):

        if tags[i][0] == "B":
            entity = tokens[i]
        # TODO: We have to know when an entity has "finished"
        #  think about the following tag sequences:
        #  B-drug -> B-drug (First entity finished)
        #  B-drug -> O (entity finished)
        #  I-drug -> I-drug (Entity not finished)
        #  I-drug -> O (Entity finished)
        #  I-drug -> B-drug (First entity finished)


if __name__ == "__main__":

    try:
        model = sys.argv[1]
        features = sys.argv[2]
    except:
        model = 'model.crfsuite'
        features = 'devel.feat'

    tagger = pycrfsuite.Tagger()
    tagger.open(model)

    with open(features, "r") as file:
        line = file.readline()

        while line:
            sentence_features = []
            sentence_tokens = []
            sid = line.split('\t')[0]
            while line != '\n':
                split = line.split('\t')
                feat = split[5:]
                sentence_features.append(feat)
                token = (split[1], split[2], split[3])
                sentence_tokens.append(token)
                line = file.readline()

            tags_pred = tagger.tag(sentence_features)
            output_entities(sid, sentence_tokens, tags_pred)
            line = file.readline()
