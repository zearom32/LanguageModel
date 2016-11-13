## This is a simple language model for Chinese short text

* naive RNN model
* LSTM


--
usage:
* Train:
```
    ## Training
    ## Every line is a sentence
    ## data should be model/ch.train.txt, model/ch.valid.txt, model/ch.test.txt
    python ptb_word_lm.py --data_path="data" --save_path="model"
```
* Test in code:
```
    ## first initialize the model with path (same as save_path)
    model = LanguageModel("model")

    ## input: X,X [['I','am','fine'], ['I','am OK']]
    ##          Chinese must be decoded as utf-8
    ## output: y, a numpy array, perplexity of each sentence

    model.test(x)

    # you can also test sentences in file
    model.test_file("test.txt")
```
