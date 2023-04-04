import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


async def inference(img, config, weight):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img', required=True, help='Path to image')
    # parser.add_argument('--config', required=True, help='Path to config file')
    # parser.add_argument('--weight', required=True, help='Path to weight file')
    # args = parser.parse_args()
    config = Cfg.load_config_from_file(config)
    config['cnn']['pretrained'] = False
    config['predictor']['beamsearch'] = False
    config['weights'] = weight
    detector = Predictor(config)

    img = Image.open(img)
    s = detector.predict(img)
    return s


if __name__ == '__main__':
    inference()
