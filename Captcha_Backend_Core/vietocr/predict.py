import argparse
from PIL import Image

from tool.predictor import Predictor
from tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to image')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weight', required=True, help='Path to weight file')
    # './vietocr/weights/transformerocr.pth'
    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    config['weights'] = args.weight
    detector = Predictor(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':
    main()
