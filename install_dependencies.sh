pip install -r requirements.txt

# install KenLM
pip install https://github.com/kpu/kenlm/archive/master.zip
mkdir  hw_asr/text_encoder/kenlm
cd hw_asr/text_encoder/kenlm
wget https://openslr.elda.org/resources/11/librispeech-vocab.txt
wget https://openslr.elda.org/resources/11/3-gram.arpa.gz
gzip -d "3-gram.arpa.gz"
cd ../../../

# dowload samples for RIR augmentation
./requirements_loaders/get_mit.sh hw_asr/augmentations/wave_augmentations

# download samples for BackgroundNoise augmentation
./requirements_loaders/get_background_noise hw_asr/augmentations/wave_augmentations

# download model
wget "https://disk.yandex.ru/d/9abt9aZHPYyEfg" -O best_model.zip
unzip best_model.zip
rm best_model.zip
