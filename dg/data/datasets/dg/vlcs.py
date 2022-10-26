import glob
import os.path as osp

from dg.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class VLCS(DatasetBase):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    dataset_dir = "VLCS"
    domains = ["caltech", "labelme", "pascal", "sun"]
    data_url = "https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd"

    def __init__(self, cfg):
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self.dataset_dir = osp.join(dataset_path, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(dataset_path, "vlcs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAIN
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "full")
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "crossval")
        test = self._read_data(cfg.DATASET.TARGET_DOMAIN, "full")

        super().__init__(train_x=train, test=test)

    def _read_data(self, input_domains, split):
        items = []
        img_id = 0
        for domain, dname in enumerate(input_domains):
            dname = dname.upper()
            path = osp.join(self.dataset_dir, dname, split)
            folders = listdir_nohidden(path)
            folders.sort()

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, "*.jpg"))

                for impath in impaths:
                    item = Datum(img_id=img_id, impath=impath, label=label, domain=domain)
                    img_id += 1
                    items.append(item)

        return items
