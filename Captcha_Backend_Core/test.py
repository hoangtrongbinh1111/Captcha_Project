import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import argparse


def test(config='config_vgg_transformer.yml', data_root='dataset/ocr/data_line', test='test_line_annotation.txt', max_lr=0.0003, weight='./vietocr/weights/vgg_transformer.pth'):
    print(args)
    config = Cfg.load_config_from_file(config)
    dataset_params = {
        'name': 'hw',
        'data_root': data_root,
        'valid_annotation': test
    }

    optim_params = {
        'max_lr': max_lr
    }
    config['cnn']['pretrained'] = False
    config['optimizer'].update(optim_params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:0'
    config['weights'] = weight
    trainer = Trainer(config, pretrained=False)
    result = trainer.test()
    return result


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description='Transfer some parameters')
    argParser.add_argument(
        "--config", default='config_vgg_transformer.yml', help="vgg_transformer or vgg_seq2seq")
    argParser.add_argument(
        "--data-root", default='dataset/ocr/data_line', help="Path to data root")
    argParser.add_argument(
        "--test", default='test_line_annotation.txt', help="test annotation")
    argParser.add_argument("--max-lr", default=0.0003,
                           type=float, help="Max learning rate")
    argParser.add_argument(
        "--weight", default='./vietocr/weights/vgg_transformer.pth', help="Path to weight")
    args = argParser.parse_args()
    test(args)
