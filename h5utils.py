from typing import Union

import h5py


def create_overwrite_dataset(
    h5_file: h5py.File,
    dataset_path: str,
    dtype,
    is_resizable: bool = True,
    **kwargs,
) -> h5py.Dataset:
    if dataset_path in h5_file:
        del h5_file[dataset_path]

    if is_resizable:
        shape = kwargs.pop("shape", (0,))
        maxshape = kwargs.pop("maxshape", (None,))
        chunks = kwargs.pop("chunks", True)
    else:
        shape = kwargs.pop("shape")
        maxshape = kwargs.pop("maxshape", None)
        chunks = kwargs.pop("chunks", None)

    new_dset = h5_file.create_dataset(
        name=dataset_path,
        shape=shape,
        maxshape=maxshape,
        dtype=dtype,
        chunks=chunks,
        **kwargs,
    )

    return new_dset
