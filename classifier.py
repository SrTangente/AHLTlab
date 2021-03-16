import sys
import pycrfsuite


def output_entities(sid, tokens, tags):
    last_tag = "O"
    entity_started = False
    i = 0
    while i < len(tags):

        # A new entity starts
        if tags[i][0] == "B":

            # Check if we have an entity already started and if so, finish it
            if entity_started:
                print(f"{sid}|{entity[1]}-{entity[2]}|{entity[0]}|{last_tag[2:]}")

            entity_started = True
            entity = tokens[i]

        # The current entity continues, update it
        elif tags[i][0] == "I":

            # In case we found an I without a B that makes sense, it is treated as a B:
            # update tags and repeat the same iteration (this time like if it was a B)
            if tags[i][2:] != last_tag[2:]:
                sys.stderr.write(f"Some entity in {sid} had a different starting B: {tags}\nIt is treated as a B\n\n")
                b_tag = 'B-' + tags[i][2:]
                tags[i] = b_tag
                continue

            updated_entity = (entity[0] + ' ' + tokens[i][0], entity[1], tokens[i][2])
            entity = updated_entity


        # Finish the entity
        elif tags[i][0] == "O" and entity_started:

            print(f"{sid}|{entity[1]}-{entity[2]}|{entity[0]}|{last_tag[2:]}")
            entity_started = False

        last_tag = tags[i]
        i += 1

    # Finish the possible remaining entity (at the end of the sentence)
    if entity_started:
        print(f"{sid}|{entity[1]}-{entity[2]}|{entity[0]}|{last_tag[2:]}")


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
