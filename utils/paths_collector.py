"""Class to get global paths"""
from pathlib import Path
from utils.paths_internal import CONFIG_DIR
from utils.configurable import Configurable


class Paths(Configurable):

    def __init__(self):
        self.config_path = CONFIG_DIR / 'config.ini'
        super().__init__(self.config_path)
        self.paths = dict()
        self._build()

    def _build(self):
        warehouse = Path(self.config['data']['warehouse'])
        self.paths['warehouse'] = warehouse
        self.paths['logs'] = Path(self.config['data']['logs'])
        self.paths['carrada'] = warehouse / 'Carrada'
        self.paths['config'] = str(self.config_path)

    def get(self):
        return self.paths
