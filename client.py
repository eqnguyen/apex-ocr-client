import argparse
import logging
import signal
import time
from datetime import datetime

import requests
from apex_ocr.engine import ApexOCREngine, SummaryType
from apex_ocr.roi import scale_rois
from apex_ocr.utils import get_primary_monitor
from config import DATA_DIRECTORY, LOG_DIRECTORY
from PIL import Image, ImageGrab
from rich.logging import RichHandler

logging.captureWarnings(True)
logger = logging.getLogger("apex_ocr.client")

# TODO: Remove this and get from Apex ROI
PRIMARY_MONITOR = get_primary_monitor()
PRIMARY_BBOX = (
    PRIMARY_MONITOR.x,
    PRIMARY_MONITOR.y,
    PRIMARY_MONITOR.x + PRIMARY_MONITOR.width,
    PRIMARY_MONITOR.y + PRIMARY_MONITOR.height,
)

RUNNING = True


def signal_handler(_signo, _stack_frame):
    global RUNNING
    RUNNING = False
    logger.info("Exit signal detected!")


def main(interval: int, num_images: int, url: str, debug: bool):
    # Initialize Apex OCR engine
    ocr_engine = ApexOCREngine()
    logger.info("Initialized Apex OCR engine")

    # Scale ROIs to primary monitor
    scale_rois()
    logger.info("Scaled ROIs")

    while RUNNING:
        # Get summary type of current screen
        logger.debug("Detecting summary page on screen...")
        summary_type = ocr_engine.classify_summary_page(debug=debug)

        if summary_type == SummaryType.SQUAD:
            logger.info("Squad summary page detected")

            # Take multiple screenshots of the results screen
            dup_images = []
            for _ in range(num_images):
                dup_images.append(ImageGrab.grab(bbox=PRIMARY_BBOX))
                time.sleep(0.5)

            # Composite the images together
            composite_image = dup_images[0]
            mask = Image.new("L", composite_image.size, 128)
            for i in range(1, num_images):
                composite_image = Image.composite(composite_image, dup_images[i], mask)

            # Save composited screenshot in Windows steam screenshot format YYYYMMDDHHmmss_1.png
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            filepath = DATA_DIRECTORY / (timestamp + "_1.png")
            composite_image.save(filepath)

            if debug:
                for i, img in enumerate(dup_images):
                    filepath = DATA_DIRECTORY / (timestamp + f"_{i+2}.png")
                    img.save(filepath)

            # Send to API endpoint
            files = {"file": open(filepath, "rb")}

            response = requests.post(url, files=files)
            logger.info("Sent file to server")
            logger.debug(response.json())

        time.sleep(interval)


if __name__ == "__main__":
    # Configure logger
    file_handler = logging.FileHandler(
        LOG_DIRECTORY
        / f"apex_ocr_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(
        level=logging.INFO,
        format=" %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            file_handler,
            RichHandler(omit_repeated_times=False, rich_tracebacks=True),
        ],
    )

    # Configure argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interval", default=3, type=int)
    parser.add_argument("-n", "--num-images", default=5, type=int)
    parser.add_argument(
        "-u", "--url", default="http://localhost:8000/uploadfile/", type=str
    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    # Connect SIGTERM and SIGINT to signal handler
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        main(args.interval, args.num_images, args.url, args.debug)
    except Exception as e:
        logger.exception(e)
