"""Lectura bajo demanda de los archivos HDF5 del Laboratorio 2."""

import h5py
import numpy as np
from torch.utils.data import Dataset


class CaudalDataset(Dataset):
    """Dataset HDF5 con splits train/validation y acceso bajo demanda."""

    SPLITS = {"train": 0, "validation": 1}

    def __init__(self, ruta, split=None):
        self.ruta = ruta
        self.split = split
        self._archivo = None

        with h5py.File(self.ruta, "r") as f:
            total = f["X"].shape[0]

            if split in self.SPLITS and "split" in f:
                self.ids = np.flatnonzero(f["split"][:] == self.SPLITS[split])
            else:
                self.ids = np.arange(total, dtype=np.int64)

    def _open(self):
        if self._archivo is None:
            self._archivo = h5py.File(self.ruta, "r")
        return self._archivo

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, indice):
        f = self._open()
        fila = int(self.ids[indice])

        muestra = {
            "X": f["X"][fila].astype(np.float32),
        }

        if "y" in f:
            muestra["y"] = f["y"][fila].astype(np.float32)

        if "y_aux" in f:
            muestra["y_aux"] = f["y_aux"][fila]

        if "Id" in f:
            muestra["Id"] = f["Id"][fila]
        elif "basin_id" in f:
            muestra["Id"] = f["basin_id"][fila]
        else:
            muestra["Id"] = fila

        return muestra
