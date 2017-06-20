#!/usr/bin/zsh
#set -x
audio=`find ~/data/deepspeech_data/wav -name "$1*".wav -size +500 | shuf -n 1 | xargs basename`
echo $audio
transcript=${audio:0:-4}.txt
python predict_2conv.py --model_path models-merged-1149-5x1024/deepspeech_13.pth.tar --audio_path ~/data/deepspeech_data/wav/$audio --transcript_path ~/data/deepspeech_data/stm/$transcript --debug
echo -en "\033[32m"
cat ~/data/deepspeech_data/stm/$transcript
echo -en "\033[0m"
