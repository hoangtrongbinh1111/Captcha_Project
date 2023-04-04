import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import argparse


async def train(config='vgg_transformer.yml', data_root='./dataset/ocr/data_line/', train='train_line_annotation.txt', test='test_line_annotation.txt', num_epochs=20000, batch_size=32, max_lr=0.0003, export='./weights/transformerocr.pth', checkpoint='./checkpoint/transformerocr_checkpoint.pth'):
    config = Cfg.load_config_from_file(config)
    dataset_params = {
        'name': 'hw',
        'data_root': data_root,
        'train_annotation': train,
        'valid_annotation': test
    }

    params = {
        'print_every': 200,
        'valid_every': 200,
        'iters': num_epochs,
        'checkpoint': checkpoint,
        'export': export,
        'batch_size': batch_size,
        'metrics': 10000
    }
    optim_params = {
        'max_lr': max_lr
    }
    config['cnn']['pretrained'] = True
    config['optimizer'].update(optim_params)
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:0'
    trainer = Trainer(config, pretrained=True)
    # trainer.config.save('config_{0}.yml'.format(config))
    train_output = trainer.train()
    for res_per_epoch in train_output:
        yield res_per_epoch
    # trainer.visualize_prediction()
    # print(trainer.precision())


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description='Transfer some parameters')
    argParser.add_argument("--config", default='vgg_transformer',
                           help="vgg_transformer or vgg_seq2seq")
    argParser.add_argument(
        "--data-root", default='./dataset/ocr/data_line/', help="Path to data root")
    argParser.add_argument(
        "--train", default='train_line_annotation.txt', help="train annotation")
    argParser.add_argument(
        "--test", default='test_line_annotation.txt', help="test annotation")
    argParser.add_argument("--num-epochs", default=20000,
                           type=int, help="Num of epochs")
    argParser.add_argument("--batch-size", default=32,
                           type=int, help="Batch size")
    argParser.add_argument("--max-lr", default=0.0003,
                           type=float, help="Max learning rate")
    argParser.add_argument(
        "--export", default='./weights/transformerocr.pth', help="Export weight")
    argParser.add_argument(
        "--checkpoint", default="./checkpoint/transformerocr_checkpoint.pth", help="Path to checkpoint")
    args = argParser.parse_args()
    # print(args)
    train(args)
