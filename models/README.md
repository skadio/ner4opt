# Ner4Opt Models

There are **four** main models in this repository:

1. Lexical Model: Uses lexical features like POS tags, Named entities etc., followed by a Conditional Random Field model.
2. Lexical Plus Model: In addition to the lexical features, we engineered more optimization specific features followed by a Conditional Random Field model.
3. Semantic Model: Semantic model is a RoBERTa-based deep learning model fine-tuned on our training data. The model is hosted on [Huggingface Models: skadio/ner4opt-roberta-v1](https://huggingface.co/skadio/ner4opt-roberta-v1) 
4. (Our best) **Hybrid Model**: This is the best performing model from our paper ([Constraints'24](https://link.springer.com/article/10.1007/s10601-024-09376-5), [CPAIOR'23](https://github.com/skadio/ner4opt/blob/main/docs/%5BCPAIOR%202023%5D%20Ner4Opt%20Paper.pdf), [NeurIPS'22](https://github.com/skadio/ner4opt/blob/main/docs/%5BNeurIPS%202022%5D%20Ner4Opt%20Poster.pdf)) that combines feature engineering and semantic feature learning models to get best results. The model is hosted on [Huggingface Models: skadio/ner4opt-roberta-v2](https://huggingface.co/skadio/ner4opt-roberta-v2). The only difference between the v1 and v2 versions of our semantic models is their _inherent randomness during training_ and we distinguish between them for reproducibility purposes. 

# Model Training
Please refer to our [training section](https://github.com/skadio/ner4opt/tree/main/models/training) to understand how our best model is trained. 

We use **Conditional Random Field** as our final layer for our Entity Recognition.

The `models` folder has the CRF models required for Entity Recognition.

## Directory Structure
```
├── lexical.pkl         <- Lexical CRF Model 
├── lexical_plus.pkl    <- Lexical Plus CRF Model
├── hybrid.pkl          <- Hybrid CRF Model
```
