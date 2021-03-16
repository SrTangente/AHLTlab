import sys
from os import listdir
from xml.dom.minidom import parse
from Baseline import tokenize, load_drug_bank
import re


def get_suffixes(word):
    suffixes = ['azole', 'idine', 'amine', 'mycin', 'asone', 'bicin', 'afil', 'bital',
                'caine', 'cillin', 'cycline', 'dipine', 'dronate', 'eprazole', 'fenac',
                'floxacin', 'gliptin', 'glitazone', 'iramine', 'ine', 'mab', 'lamide', 'mustine',
                'mycin', 'nacin', 'olol', 'odone', 'olone', 'onide', 'parin', 'phylline', 'pril',
                'profen', 'ridone', 'sartan', 'semide', 'setron', 'statin', 'terol',
                'thiazide', 'tinib', 'trel', 'tretin', 'triptan', 'vir', 'vudine', 'zepam',
                'zolam', 'zosin']
    for suff in suffixes:
        if word.endswith(suff):
            return suff
    return None


def get_prefixes(word):
    prefixes = ['cef', 'ceph', 'cort', 'pred', 'sulfa']
    for pr in prefixes:
        if word.startswith(pr):
            return pr
    return None


def get_tag(token, gold):
    for triplet in gold:
        if token[1] == triplet[0]:
            return 'B-' + triplet[2]
        elif token[1] >= triplet[0] and token[2] <= triplet[1]:
            return 'I-' + triplet[2]
    return 'O'

def append_features(feature_i, word):
    prefix = get_prefixes(word)
    suffix = get_suffixes(word)
    if prefix:
        feature_i.append("pref=" + prefix)
    if suffix:
        feature_i.append("suff=" + suffix)
    if re.search('^.*[0-9]+.*$', word) is not None:
        feature_i.append("hasDigits=" + '1')
    if re.search('^.*-+.*$', word) is not None:
        feature_i.append("hasHyphen=" + '1')
    if re.search('^.*,+.*$', word) is not None:
        feature_i.append("hasComma=" + '1')
    if re.search('^.*[({]+.*$', word) is not None or re.search('^.*[})]+.*$', word):
        feature_i.append("hasParenthesis=" + '1')
    if re.search('^.*\]+.*$', word) is not None or re.search('^.*\[+.*$', word) is not None:
        feature_i.append("hasParenthesis=" + '1')
    if re.search('^.*\%+.*$', word) is not None:
        feature_i.append("hasPercent=" + '1')
    if word.isupper():
        feature_i.append("upper=" + '1')
    elif word[0].isupper():
        feature_i.append("capitalized=" + '1')

def extract_features(toks, drug_bank):
    result = []
    for i in range(len(toks)):
        word = toks[i][0]
        next_word = None
        prev_word = None
        if i > 0:
            prev_word = toks[i - 1][0]
        if i < len(tokens) - 1:
            next_word = toks[i + 1][0]

        feature_i = ["form=" + word]
        append_features(feature_i, word)

        if prev_word:
            feature_i.append("prev=" + prev_word)
        if next_word:
            feature_i.append("next=" + next_word)


        lower_word = word.lower()
        if lower_word in drug_bank:
            feature_i.append("external="+drug_bank[lower_word])
        if prev_word is not None and prev_word.lower() in drug_bank:
            feature_i.append("prevExternal="+drug_bank[prev_word.lower()])
        if next_word is not None and next_word.lower() in drug_bank:
            feature_i.append("nextExternal="+drug_bank[next_word.lower()])

        result.append(feature_i)

    return result

if __name__ == "__main__":

    try:
        datadir = sys.argv[1]
    except:
        datadir = "data/train/"

    drug_bank = load_drug_bank()

    # process each file in directory
    for f in listdir(datadir):
        # parse XML file , obtaining a DOM tree
        tree = parse(datadir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # load ground truth entities .
            gold = []
            entities = s.getElementsByTagName("entity")
            for e in entities:
                # for discontinuous entities , we only get the first span
                offset = e.attributes["charOffset"].value
                (start, end) = offset.split(";")[0].split("-")
                gold.append((int(start), int(end), e.attributes["type"].value))
            # tokenize text
            try:
                tokens = tokenize(stext)
            except ValueError:
                #print(f"Omitting '{stext}' because a ValueError")
                continue
            # extract features for each word in the sentence
            features = extract_features(tokens, drug_bank)
            # print features in format suitable for the learner / classifier
            for i in range(0, len(tokens)):
                # see if the token is part of an entity , and which part (B/I)
                tag = get_tag(tokens[i], gold)
                print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')
            # blank line to separate sentences
            print()
