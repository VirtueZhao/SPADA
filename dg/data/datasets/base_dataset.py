import os

import os.path as osp
import tarfile
import zipfile

import gdown

from dg.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        dname (str): domain name.
        classname (str): class name.
    """
    def __init__(self, img_id, impath="", label=0, domain=0, dname="", classname="", augment_flag=True):
        assert isinstance(impath, str)
        assert check_isfile(impath)
        self._img_id = img_id
        self._impath = impath
        self._label = label
        self._domain = domain
        self._dname = dname
        self._classname = classname
        self._augment_flag = augment_flag
        self._history_info = []

    @property
    def img_id(self):
        return self._img_id

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def dname(self):
        return self._dname

    @property
    def classname(self):
        return self._classname

    @property
    def augment_flag(self):
        return self._augment_flag

    @property
    def history_info(self):
        return self._history_info

    def set_augment_flag(self, augment_flag):
        self._augment_flag = augment_flag

    def set_history_info(self, history_info):
        self._history_info.append(history_info)


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""                    # the directory where the dataset is stored
    domains = []                        # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        print("+Calling: DDAIG.__init__().SimpleTrainer.__init__().build_data_loader().DataManager.__init__().build_dataset().PACS.__init__().DatasetBase.__init__()")
        self._train_x = train_x         # labeled training data
        self._train_u = train_u         # unlabeled training data (optional)
        self._val = val                 # validation data (optional)
        self._test = test               # test data

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_label_to_classname(train_x)
        print("-Closing: DDAIG.__init__().SimpleTrainer.__init__().build_data_loader().DataManager.__init__().build_dataset().PACS.__init__().DatasetBase.__init__()")

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_label_to_classname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domain):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domain)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print("File extracted to {}".format(osp.dirname(dst)))
