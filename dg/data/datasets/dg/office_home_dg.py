import os.path as osp

from ..build import DATASET_REGISTRY
from .digits_dg import DigitsDG
from ..base_dataset import DatasetBase


@DATASET_REGISTRY.register()
class OfficeHomeDG(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg):
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self.dataset_dir = osp.join(dataset_path, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(dataset_path, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAIN
        )

        train = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "all"
        )
        val = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "val"
        )
        test = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAIN, "all"
        )

        super().__init__(train_x=train, test=test)
