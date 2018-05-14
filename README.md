# Semantic Textual Similarity
## Author: Shiva Suri (github: shivasuri)

## Run Instructions & Prerequisites
Download google’s word embeddings (word2vec). Make sure the GoogleNews-vectors-negative300.bin file you have downloaded is in the same directory as the code (i.e. main.py).
Make sure you have installed the nltk library and WordNet; I use WordNet to find synonyms and antonyms.
>> conda install nltk
>> nltk.download()

In the pop-up window, select to download the “all” library.

Please be sure to install all libraries included as imports in the top of main.py
Make sure data/ contains en-train.txt, en-val.txt, and en-test.txt as provided.

To run my code, type in command line:
>> python main.py --inputfile [data/en-val.txt | data/en-test.txt]

This will run three functions, which correspond to the baseline and two extensions. The baseline and first extension will yield in two files: pred_simple.txt and pred_ex1.txt, respectively. Each of the two files contains predictions on the sentence pairs provided in the input file. However, the second extension automatically yields in two of its own files, for validation and test sets, and the outputs are in pred_val_ex2.txt and pred_test_ex2.txt.

Please be patient when running the script, as the supervised model in Extension 2 takes O(1.5min.) to run.

To evaluate predictions, run evaluate.py with
>> python evaluate.py --predfile [pred….txt] --goldfile [data/en-val.txt | data/en-test.txt]

## Credits
Cer et al.: http://www.aclweb.org/anthology/S17-2001
Maharjan et al.: http://www.aclweb.org/anthology/S17-2014
word2vec
WordNet
