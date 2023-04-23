# Models

There are **four** main models in this repository

1. Lexical Model - Uses lexical features like POS tags, Named entities etc., followed by a Conditional Random Field model
2. Lexical Plus Model - In addition to the lexical features, we engineered more optimization specific features followed by a Conditional Random Field model
3. Semantic Model - Semantic model is a RoBERTa-based deep learning model fine-tuned on our training data. You can find the (v1 version) model [here](https://huggingface.co/skadio/ner4opt-roberta-v1) 
4. (Our best) **Hybrid Model** - This model combines the above feature engineering and "semantic" feature learning models to get best results. 
Please refer to our [training section](https://github.com/skadio/Ner4Opt/training) to understand how our best model is trained. 
Our "Semantic Model (v2 version)" trained as a part of our hybridization experiment can be found [here](https://huggingface.co/skadio/ner4opt-roberta-v2)

_The only difference between the v1 and v2 versions of our semantic models is their inherent randomness during training_

We use **Conditional Random Field** as our final layer for our Entity Recognition.

The `models` folder has the CRF models required for Entity Recognition.

## Directory structure
```
├── lexical.pkl         <- Lexical CRF Model 
├── lexical_plus.pkl    <- Lexical Plus CRF Model
├── hybrid.pkl          <- Hybrid CRF Model
```
