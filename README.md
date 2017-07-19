# dbcollection for Torch7

This is a simple Lua wrapper for the Python's [dbcollection](https://github.com/farrajota/dbcollection) module. The functionality is almost the same, appart from some few minor differences related to Lua, namely regarding setting up ranges when fetching data.

Internally it calls the Python's dbcollection module for data download/process/management. The, Lua/Torch7 interacts solely with the metadata `hdf5` file to fetch data from disk.

## Package installation

### Requirements

This package requires:

- Python's dbcollection package installed.
- [Torch7](https://github.com/torch/torch7)
- json
- argcheck

To install Torch7 just follow the steps listed [here](http://torch.ch/docs/getting-started.html#_).

The other packages should come pre-installed along with Torch7, but in case they don't, you can simply install them by doing the following:

```lua
luarocks install json
luarocks install argcheck
```

### Installation

To install the dbcollection's Lua/Torch7 API, first the Python's version must be installed on your system. If you do not have it already installed, then you can install it either via `pip`, `conda` or from [source](https://github.com/dbcollection/dbcollection#package-installation). Here we'll use `pip` to install this package:

```
pip install dbcollection
```

Then, all there is to do is to clone this repo and install the package via `luarocks`:

```
git clone https://github.com/dbcollection/dbcollection-torch7
cd dbcollection-torch7 && luarocks make
```


## Usage

This package follows the same API as the Python version. Once installed, to use the package simply require *dbcollection*:

```lua
dbc = require 'dbcollection'
```

Then, just like with the Python's version, to load a dataset you simply do:

```lua
mnist = dbc.load('mnist')
```

You can also select a specific task for any dataset by using the `task` option.

```lua
mnist = dbc.load{name='mnist', task='classification'}
```

This API lets you download+extract most dataset's data directly from its source to the disk. For that, simply use the `download()` method:

```lua
dbc.download{name='cifar10', data_dir='home/some/dir'}
```

### Data fetching

Once a dataset has been loaded, in order to retrieve data you can either use Torch7's `HDF5` API or use the provided methods to retrive data from the .h5 metadata file.

For example, to retrieve an image and its label from the `MNIST` dataset using the Torch7's `HDF5` API you can do the following:

```lua
images_ptr = mnist.file:read('default/train/images')
img = images_tr:partial({1,1}, {1,32}, {1,32}, {1,3})

labels_ptr = mnist.file:read('default/train/labels')
label = labels_ptr:partial({1,1})
```

or you can use the API provided by this package:

```lua
img = mnist:get('train', 'images', 1)
label = mnist:get('train', 'labels', 1)
```


## Documentation

For a more detailed view of the Lua's API documentation see [here](DOCUMENTATION.md#db.documentation).


## License

MIT license (see the [LICENSE](LICENSE) file)