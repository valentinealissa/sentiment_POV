# The Point of View of a Sentiment: Towards Clinician Bias Detection in Psychiatric Notes
> The aim of this work is to leverage pretrained and large language models (PLMs and LLMs) to characterize harmful language usage in psychiatry
> by identifying the sentiment expressed in sentences describing psychiatric patients based on the reader’s point of view. We fine-tuned three PLMs
> (RoBERTa, GatorTron, and GatorTron + Task Adaptation) and implemented zero-shot/few-shot in-context learning (ICL) approaches for three LLMs
> (GPT-3.5, Llama-3.1, and Mistral) to classify the sentiment of the sentences according to the physician or non-physician point of view.
> Our project underlines the importance of recognizing the reader’s point of view, not only for regulating the note writing process,
> but also for the quantification, identification, and reduction of bias in computational systems for downstream analyses.
## Data
For this project, we extracted 39 sentences from the Mount Sinai Health System containing psychiatric lexicon. 
Physicians (N = 10) and non-physicians (N = 10) acted as annotators. 
Physician raters have medical degrees and extensive experience writing clinical notes. 
Non-physician raters have no clinical experience, nor medical degree. 
Physicians were given annotation instructions: “If you're the physician who wrote this sentence: what is your attitude towards the patient?”
Non-physicians were given annotation instructions: “If you're the patient: how do you feel reading this description of you?”
Annotaiton data is available upon request.
## Models
|Model|Pre-training data|Fine-tuning data|Parameters|
|-----|-----------------|----------------|----------|
|Twitter-roBERTa-base for Sentiment Analysis|~124M tweets and English words|TweetEval benchmark|125 million|
|GatorTron-base|>90 billion words from clinical notes and biomedical text|NA|345 million|
|GatorTron-base-TA|GatorTron-base task adapted on 433 sentences about psych patients; Epochs = 5,  Batch Size = 4|NA|345 million|
|GPT-3.5-turbo|>570 GB of text|Undisclosed|Undisclosed|
|Llama-3.1-8B-Instruct|15 trillion tokens|Dialogue use cases with and SFT and RLHF|8 billion|
|Mistral-7B-Instruct-v0.2|Undisclosed|Instruction and conversation datasets|7 billion|
## Classification Approach with PLMs
The classification task was designed to ask the models to classify the corresponding sentiment of the sentences from the physician and non-physician point of view. 
The seed of the models was set to 42.
### Task adaptation
We task adapted GatorTron on sentences written about psychiatric patients (N = 433; Epoch = 5; Batch size = 4), resulting in the GatorTron-TA model.
### Fine-tuning
Each model was fine-tunned to label the sentiment of sentences in each training set for the physician and non-physician labels.
Hyperparameter optimization was performed during fine-tuning of each model using a grid search approach. 
Key hyperparameters included learning rate, batch size, and the number of epochs, tested within ranges: learning rate (5e-6 to 1e-4), batch size (2 to 6), epochs (1 to 4), weight decay (0 to 0.1), warm up ratio (0 to 0.1). 
Each configuration was evaluated on macro F1 score, with early stopping applied to prevent overfitting. 
## Prompt-based Approach with LLMs
Prompts were designed to ask the model to classify the corresponding sentiment of the sentences from the physician and non-physician point of view. 
The temperature of the models was set to 0 or 0.001 and seed to 42.
### Zero Shot Prompt
Our ICL prompt-based approach utilized two prompts, one for the physician task and one for the non-physician task, for each model.
### ICL Prompt
Our ICL prompt-based approach utilized two prompts, one for the physician task and one for the non-physician task, for each model. 
In each prompt, we utilized the subsets of training sentences as contextual examples for the model. 
To determine which training sentences led to the best model’s performance, we engineered our prompt using different combinations of negative, neutral, and positive sentences. 
For example, the training sentences with 80% agreement contained 2 negative, 3 neutral, and 2 positive sentences, so we created a matrix to loop through every combination: 
```
[negative:[0,1,2];neutral: [0,1,2,3]; positive:[0,1,2]]
```


