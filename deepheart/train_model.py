from parser import PCG
from model import CNN
import sys


true_strs = {"True", "true", "t"}


def load_and_train_model(model_path, load_pretrained, pre_data_path="/tmp", log_path="/tmp"):
    pcg = PCG(model_path)

    if load_pretrained:
        pcg.load(pre_data_path)
    else:
        pcg.initialize_wav_data(pre_data_path)

    cnn = CNN(pcg, epochs=1000, dropout=0.5, base_dir=log_path)
    cnn.train()

if __name__ == '__main__':
    data_path = sys.argv[1]

    load_pretrained = False
    if len(sys.argv) == 3:
        load_pretrained = sys.argv[2] in true_strs

    load_and_train_model(data_path, load_pretrained)