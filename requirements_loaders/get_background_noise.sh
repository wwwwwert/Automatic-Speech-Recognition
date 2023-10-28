dest=$1

cd $dest
git clone https://github.com/Rumeysakeskin/Speech-Data-Augmentation.git --depth 1
mv -if Speech-Data-Augmentation/background_noise ./
rm -rf Speech-Data-Augmentation
