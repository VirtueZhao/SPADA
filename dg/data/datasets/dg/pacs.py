import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class PACS(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """

    dataset_dir = "pacs"
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg):
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().build_data_loader().DataManager.__init__().build_dataset().PACS.__init__()")
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self.dataset_dir = osp.join(dataset_path, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        if not osp.exists(self.dataset_dir):
            dst = osp.join(dataset_path, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAIN
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "all")
        # val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "crossval")
        test = self._read_data(cfg.DATASET.TARGET_DOMAIN, "all")

        super().__init__(train_x=train, test=test)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().build_data_loader().DataManager.__init__().build_dataset().PACS.__init__()")

    def _read_data(self, input_domains, split):
        items = []
        img_id = 0
        for domain, dname in enumerate(input_domains):
            if split == "all":
                file_train = osp.join(self.split_dir, dname + "_train_kfold.txt")
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval_kfold.txt")
                impath_label_list += self._read_split_pacs(file_val)

            else:
                file = osp.join(self.split_dir, dname + "_" + split + "_kfold.txt")
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                item = Datum(
                    img_id=img_id,
                    impath=impath,
                    label=label,
                    domain=domain,
                    dname=dname,
                    classname=classname
                )
                img_id += 1
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items
