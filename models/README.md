# Models

Please refer [training]() section to understand how our best model is trained

There are **three** main models in this repository

1. Lexical Model - Uses lexical features like POS tags, Named entities etc., followed by a Conditional Random Field model
2. Lexical Plus Model - In addition to the lexical features, we engineered more optimization specific features followed by a Conditional Random Field model
3. (Our best) **Hybrid Model** - In addition to lexical features, engineered features, we added transformer predictions as a feature followed by a Conditional Random Field model

## Directory structure
```
├── lexical.pkl         <- Lexical CRF Model 
├── lexical_plus.pkl    <- Lexical Plus CRF Model
├── hybrid.pkl          <- Hybrid CRF Model
```
