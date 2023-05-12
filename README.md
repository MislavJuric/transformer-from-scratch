# README

This is my implementation of Transformers from scratch (in PyTorch). Transformers were introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). My goal was to implement the model described in the paper without looking at any other existing implementation(s). My goal was to ensure that the model was training correctly and that, once saved and loaded, it produced correct results. I did not aim to reproduce the results from the paper, nor to implement all of the bells and whistles; when I was sure the model was training correctly and that I could use the trained model correctly, I ceased development.

I would also like to note that, since I was focused on translating concepts from the paper to code, the code I have written can most probably be written more efficiently (and also more readably as well). I have noted some potential improvement areas with `TODO`s in my code and other improvements in **Future_work.txt**.
 
## Files and folders description

You can download the Transformer model weights (which were trained on the **mini_train** dataset) for 2048 epochs [here](https://drive.google.com/file/d/1bT2rt1qBustTHrkjP0FGFSj5w9NRJTIs/view?usp=sharing). **mini-train** dataset was obtained by taking the first 7 sentences from English-German dataset found [here](https://nlp.stanford.edu/projects/nmt/).

You can re-create my virtual environment in [(Ana)conda](https://www.anaconda.com/) by running:

```
conda env create -f environment.yml
```

The list below contains the description of files and folders in this repository that I think are relevant. *Note*: If you see a **deprecated** folder within any of the subfolders, that's a particular file I have deprecated for the reason noted at its beginning. 

 - **blocks** - contains the implementations of the Encoder block and the Decoder block
 - **dataset** - contains the **mini_dataset** which I used to test if my model was training properly
 - **deprecated** - code that is deprecated for some reason (that reason is added as a comment at the beginning of the file)
 - **environment.yml** - a file which you can use to re-create my virtual environment, as described at the beginning of this section
 - **Future_work.txt** - an unsorted list of things which could be implemented in the future in order to improve the model (and/or add new features to it)
 - **juypter_notebooks** - contains a Jupyter notebook I used to visualize training loss over epochs
 - **layers** - contains the implementation of all the layers used in the Transformer model (such as Scaled-Dot Product Attention, Multi-Head Attention etc.)
 - **models** - contains the implementation of the Transformer, as well as Transformer Encoder and Transformer Decoder (which make up the Transformer)
 - **README.md** - this file
 - **script_outputs** - contains outputs from the scripts in the root repository folder (descirbed below)
 - **tests** - contains tests that I used to check I haven't got any obvious errors in my code (such as shape errors etc.)
 - **test_trained_model_on_entire_target_sequence.py** - tests a trained Transformer model on entire target sequence
 - **test_trained_model_token_by_token_on_generated_tokens.py** - tests a trained Transformer model token by token; the next token is the most probable one
 - **test_trained_model_token_by_token_on_true_tokens.py** - tests a trained Transformer model token by token; the next token is the true next token
 - **test_untrained_model_baseline.py** - tests an untrained Transformer token by token; this was my baseline
 - **Time_log.txt** - my own notes about how much time I spent doing what
 - **train.py** - the training script
 - **utils** - contains the Transformer dataset implementation, as well as the positional encoding function implementation
