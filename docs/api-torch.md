# `pyddstore.torch` reference

Generated from the docstrings. For how the pieces fit together, see
[PyTorch integration](pytorch.md).

```{eval-rst}
.. automodule:: pyddstore.torch
   :no-members:

.. autoclass:: pyddstore.torch.DistDataset(source, name, comm=None, ddstore_width=None, device=None, add_device=None, method=None, handshake_dir=None, chunk_size=None, encode=None, decode=None, fields=None)
   :members: read_rows, alloc, release, shapes, dtypes, __getitems__

.. autoclass:: pyddstore.torch.DistDatasetReader(name, handshake_dir=None, n_core=None, device=None, decode=None)
   :members: read_rows, alloc, release, shapes, dtypes, __getitems__

.. autoclass:: pyddstore.torch.WindowedDataset(ds, window, stride=1, dilation=1, starts=None, fields=None)

.. autofunction:: pyddstore.torch.row_of

.. autoclass:: pyddstore.torch.ThreadDataLoader(dataset, reuse_buffers=False, collate_copies=False, **DataLoader_kwargs)
   :members: close
```
