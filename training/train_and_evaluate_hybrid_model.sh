python3.9 -m venv ./env
nvidia-smi
uname -r
source env/bin/activate
python --version
pip install wheel setuptools
pip install allennlp overrides inflect nltk
python -m nltk.downloader all
pip install simpletransformers==0.63.7
pip install spacy==3.2.0
pip install spacy-transformers==1.1.7
python -m spacy download en_core_web_sm==3.2.0
python -m spacy download en_core_web_trf==3.2.0
pip install git+https://github.com/MeMartijn/updated-sklearn-crfsuite.git#egg=sklearn_crfsuite
pip install torch==1.12.0
current_dir=$(pwd)
previous_dir=$(dirname "$current_dir")
python train_and_evaluate_hybrid_model.py --train "$previous_dir/data/train.txt" --test "$previous_dir/data/dev.txt"
