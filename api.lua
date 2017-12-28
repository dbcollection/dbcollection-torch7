--[[
    dbcollection managing functions.
--]]

require 'paths'
local dbcollection = require 'dbcollection.env'
require 'dbcollection.loader'

local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'
local json = require 'json'


-----------------------------------------------------------
-- utility functions
-----------------------------------------------------------

function os.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end

-- cache file path
local function get_cache_file_path(options)
    local home_dir = os.getenv('HOME')
    if options.is_test then
        return paths.concat(home_dir, 'tmp', 'dbcollection.json')
    else
        return paths.concat(home_dir, 'dbcollection.json')
    end
end

--[[ Load the cache file into memory ]]
local function load_cache(options)
    local home_path = get_cache_file_path(options)
    if not paths.filep(home_path) then
        dbcollection.config_cache({is_test=options.is_test})
    end
    return json.load(home_path)
end

--[[ parse all booleans to strings in python format ]]
local function tostring_py(input)
    assert(type(input)=='boolean')
    if input then
        return 'True'
    else
        return 'False'
    end
end

local function tostring_py2(input)
    if input == 'true' then
        return 'True'
    elseif input == 'false' then
        return 'False'
    else
        return ('\'%s\''):format(input)
    end
end


--[[ convert to string if it does not match "None" ]]
local function tostring_none(input)
    if string.match(input, "None") then
        return input
    else
        return ('\'%s\''):format(input)
    end
end


--[[ get the dataset's data and cache paths ]]
local function get_dataset_paths(cache, name, task)
    local data_dir = cache['dataset'][name]['data_dir']
    local cache_path = cache['dataset'][name]['tasks'][task]
    return data_dir, cache_path
end


-- check if the task exists in the cache
local function exists_task(cache, name, task)
    if next(cache['dataset'][name]['tasks']) then
        if cache['dataset'][name]['tasks'][task] then
            return true
        else
            return false
        end
    else
        return false
    end
end

--[[ check if a task exists in cache for a dataset ]]
local function exists_task_in_cache(options)
    assert(options)
    local cache = load_cache(options)
    return exists_task(cache, options.name, options.task)
end

--[[ Return the correct name of the default task for a dataset. ]]
local function get_default_task_name(name)
    local cmd = 'from dbcollection.manager import get_default_task_name;' ..
                ('print(get_default_task_name(\'%s\'))'):format(name)
    return os.capture(string.format('python -c "%s"', cmd))
end

--[[ Validates / corrects the task name ]]
local function validate_task_name(options)
    assert(options)
    if options.task == 'default' then
        options.task= get_default_task_name(options.name)
    end
end

--[[ Check if the dataset records exist in the cache ]]
local function is_dataset_in_cache(options)
    local cache = load_cache(options)
    if cache['dataset'][options.name] then
        return true
    else
        return false
    end
end

--[[ Download a dataset's data files if there are no records in the cache file ]]
local function download_data(options)
    if not is_dataset_in_cache(options) then
        dbcollection.download({name=options.name,
                               data_dir=options.data_dir,
                               extract_data=true,
                               verbose=options.verbose,
                               is_test=options.is_test})
    end
end

--[[ Processes the dataset's metadata if there are no records in the cache file ]]
local function process_data(options)
    if not exists_task_in_cache(options) then
        dbcollection.process({name=options.name,
                              task=options.task,
                              verbose=options.verbose,
                              is_test=options.is_test})
    end
end

--[[ load check if task exists after download + process setup ]]
local function check_if_task_exists(options)
    assert(options)
    if not exists_task_in_cache(options) then
        error('Dataset name/task not available in cache for load.')
    end
end

--[[ Returns a data loader for a dataset ]]
local function get_data_loader(options)
    assert(options)
    local cache = load_cache(options)
    local data_dir, cache_path = get_dataset_paths(cache, options.name, options.task)
    return dbcollection.DatasetLoader(options.name, options.task, data_dir, cache_path)
end

--[[ Get all datasets into a table ]]
function fetch_list_datasets()
    local db_list = {}
    return db_list
end

--------------------------------------------------------------------------------
-- Vars
--------------------------------------------------------------------------------

dbcollection.available_datasets_list = fetch_list_datasets()


-----------------------------------------------------------
-- API functions
-----------------------------------------------------------

function dbcollection.download(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Download a dataset data to disk.

            This method will download a dataset's data files to disk. After download,
            it updates the cache file with the  dataset's name and path where the data
            is stored.

            Parameters
            ----------
            name : str
                Name of the dataset.
            data_dir : str, optional
                Directory path to store the downloaded data.
            extract_data : bool, optional
                Extracts/unpacks the data files (if true).
            verbose : bool, optional
                Displays text information (if true).
            is_test : bool, optional
                Flag used for tests.

            Examples
            --------
            Download the CIFAR10 dataset to disk.

            >>> dbc = require 'dbcollection'
            >>> dbc.download('cifar10')

        ]],
        {name="name", type="string",
         help="Name of the dataset."},
        {name="data_dir", type="string", default='None',
         help="Directory path to store the downloaded data.",
         opt = true},
        {name="extract_data", type="boolean", default=true,
         help="Extracts/unpacks the data files (if true).",
         opt = true},
        {name="verbose", type="boolean", default=true,
         help="Displays text information (if true).",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    local command = ('import dbcollection as dbc;' ..
                    'dbc.download(name=\'%s\',data_dir=%s,extract_data=%s,verbose=%s,is_test=%s)')
                    :format(args.name,
                            tostring_none(args.data_dir),
                            tostring_py(args.extract_data),
                            tostring_py(args.verbose),
                            tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end

------------------------------------------------------------------------------------------------------------

function dbcollection.process(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Process a dataset's metadata and stores it to file.

            The data is stored a a HSF5 file for each task composing the dataset's tasks.

            Parameters
            ----------
            name : str
                Name of the dataset.
            task : str, optional
                Name of the task to process.
            verbose : bool, optional
                Displays text information (if true).
            is_test : bool, optional
                Flag used for tests.

            Examples
            --------
            Download the CIFAR10 dataset to disk.

            >>> dbc = require 'dbcollection'
            >>> dbc.process({name='cifar10', task='classification', verbose=False})

        ]],
        {name="name", type="string",
         help="Name of the dataset."},
        {name="task", type="string", default='all',
         help="Name of the dataset.",
         opt = true},
        {name="verbose", type="boolean", default=true,
         help="Displays text information (if true).",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    assert(args.name, ('Must input a valid dataset name: %s'):format(args.name))

    local command = ('import dbcollection as dbc;' ..
                    'dbc.process(name=\'%s\',task=\'%s\',verbose=%s,is_test=%s)')
                    :format(args.name,
                            args.task,
                            tostring_py(args.verbose),
                            tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end

------------------------------------------------------------------------------------------------------------

function dbcollection.load(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Returns a metadata loader of a dataset.

            Returns a loader with the necessary functions to manage the selected dataset.

            Parameters
            ----------
            name : str
                Name of the dataset.
            task : str, optional
                Name of the task to load.
            data_dir : str, optional
                Directory path to store the downloaded data.
            verbose : bool, optional
                Displays text information (if true).
            is_test : bool, optional
                Flag used for tests.

            Returns
            -------
            DataLoader
               Data loader class.

            Examples
            --------
            Load the MNIST dataset.

            >>> dbc = require 'dbcollection'
            >>> mnist = dbc.load('mnist')
            >>> print('Dataset name: ' .. mnist.db_name)
            Dataset name:  mnist

        ]],
        {name="name", type="string",
         help="Name of the dataset."},
        {name="task", type="string", default='default',
         help="Name of the task to load.",
         opt = true},
        {name="data_dir", type="string", default='',
         help="Directory path to store the downloaded data.",
         opt = true},
        {name="verbose", type="boolean", default=true,
         help="Displays text information (if true).",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true}
    }

    local args = initcheck(...)  -- parse options
    validate_task_name(args)
    download_data(args)
    process_data(args)
    check_if_task_exists(args)
    local loader = get_data_loader(args)
    return loader
end

------------------------------------------------------------------------------------------------------------

local function parse_keywords_format(keywords)
    local kwords
    if next(keywords) then
        local str = ''
        for i=1, #keywords do
            str = str .. tostring(keywords[i])
            if i<#keywords then str = str .. ',' end
        end
        kwords = ("['%s']"):format(str)
    else
        kwords = '[]'
    end
    return kwords
end

function dbcollection.add(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Add a dataset/task to the list of available datasets for loading.

            Parameters
            ----------
            name : str
                Name of the dataset.
            task : str
                Name of the task to load.
            data_dir : str
                Path of the stored data in disk.
            file_path : bool
                Path to the metadata HDF5 file.
            keywords : list of strings, optional
                List of keywords to categorize the dataset.
            is_test : bool, optional
                Flag used for tests.

            Examples
            --------
            Add a dataset manually to dbcollection.

            >>> dbc = require 'dbcollection'
            >>> dbc.add('new_db', 'new_task', 'new/path/db', 'newdb.h5', ['new_category'])
            >>> dbc.query('new_db')
            {'new_db': {'tasks': {'new_task': 'newdb.h5'}, 'data_dir': 'new/path/db', 'keywords':
            ['new_category']}}

        ]],
        {name="name", type="string",
         help="Name of the dataset."},
        {name="task", type="string",
         help="Name of the task to load."},
        {name="data_dir", type="string",
         help="Path of the stored data in disk."},
        {name="file_path", type="string",
         help="Path to the metadata HDF5 file."},
        {name="keywords", type="table", default={},
         help="List of keywords to categorize the dataset.",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    assert(args.name, ('Must input a valid dataset name: %s'):format(args.name))
    assert(args.task, ('Must input a valid dataset name: %s'):format(args.task))
    assert(args.data_dir, ('Must input a valid dataset name: %s'):format(args.data_dir))
    assert(args.file_path, ('Must input a valid dataset name: %s'):format(args.file_path))

    -- parse the table into a string in python's format
    args.keywords = parse_keywords_format(args.keywords)

    local command = ('import dbcollection as dbc;' ..
                    'dbc.add(name=\'%s\',task=\'%s\',data_dir=\'%s\',' ..
                    'file_path=\'%s\',keywords=%s,is_test=%s)')
                    :format(args.name,
                            args.task,
                            args.data_dir,
                            args.file_path,
                            args.keywords,
                            tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end

------------------------------------------------------------------------------------------------------------

function dbcollection.remove(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Remove/delete a dataset and/or task from the cache.

            Removes the datasets cache information from the dbcollection.json file.
            The dataset's data files remain in disk if 'delete_data' is set to False,
            otherwise it removes also the data files.

            Also, instead of deleting the entire dataset, removing a specific task
            from disk is also possible by specifying which task to delete. This removes
            the task entry for the dataset and removes the corresponding hdf5 file from
            disk.

            Parameters
            ----------
            name : str
                Name of the dataset to delete.
            task : str, optional
                Name of the task to delete.
            delete_data : bool, optional
                Delete all data files from disk for this dataset if True.
            is_test : bool, optional
                Flag used for tests.

            Examples
            --------
            Remove a dataset from the list.

            >>> dbc = require 'dbcollection'
            >>> -- add a dataset
            >>> dbc.add('new_db', 'new_task', 'new/path/db', 'newdb.h5', ['new_category'])
            >>> dbc.query('new_db')
            {'new_db': {'tasks': {'new_task': 'newdb.h5'}, 'data_dir': 'new/path/db',
            'keywords': ['new_category']}}
            >>> dbc.remove('new_db')  -- remove the dataset
            Removed 'new_db' dataset: cache=True, disk=False
            >>> dbc.query('new_db')  -- check if the dataset info was removed (retrieves an empty dict)
            {}

        ]],
        {name="name", type="string",
         help="Name of the dataset."},
        {name="task", type="string", default='None',
         help="Name of the task to delete."},
        {name="delete_data", type="boolean", default=false,
         help="Delete all data files from disk for this dataset if True.",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    assert(args.name, ('Must input a valid dataset name: %s'):format(args.name))

    local command = ('import dbcollection as dbc;' ..
                    'dbc.remove(name=\'%s\',task=%s,delete_data=%s,is_test=%s)')
                    :format(args.name,
                            tostring_none(args.task),
                            tostring_py(args.delete_data),
                            tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end

------------------------------------------------------------------------------------------------------------

function dbcollection.config_cache(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Configure the cache file.

            This method allows to configure the cache file directly by selecting
            any data field/value. The user can also manually configure the file
            if he/she desires.

            To modify any entry in the cache file, simply input the field name
            you want to change along with the new data you want to insert. This
            applies to any field/data in the file.

            Another thing available is to reset/clear the entire cache paths/configs
            from the file by simply enabling the 'reset_cache' flag to true.

            Also, there is an option to completely remove the cache file+folder
            from the disk by enabling 'delete_cache' to True. This will remove the
            cache dbcollection.json and the dbcollection/ folder from disk.

            Parameters
            ----------
            field : str, optional
                Name of the field to update/modify in the cache file.
            value : str, list, table, optional
                Value to update the field.
            delete_cache : bool, optional
                Delete/remove the dbcollection cache file + directory.
            delete_cache_dir : bool, optional
                Delete/remove the dbcollection cache directory.
            delete_cache_file : bool, optional
                Delete/remove the dbcollection.json cache file.
            reset_cache : bool, optional
                Reset the cache file.
            verbose : bool, optional
                Displays text information (if true).
            is_test : bool, optional
                Flag used for tests.

            Examples
            --------
            Delete the cache by removing the dbcollection.json cache file.
            This will NOT remove the file contents in dbcollection/. For that,
            you must set the *delete_cache_dir* argument to True.

            >>> dbc = require 'dbcollection'
            >>> dbc.config_cache({delete_cache_file=true})

        ]],
        {name="field", type="string", default="None",
         help="Name of the field to update/modify in the cache file.",
         opt = true},
        {name="value", type="string", default="None",
         help="Value to update the field.",
         opt = true},
        {name="delete_cache", type="boolean", default=false,
         help="Delete/remove the dbcollection cache file + directory.",
         opt = true},
        {name="delete_cache_dir", type="boolean", default=false,
         help="Delete/remove the dbcollection cache directory.",
         opt = true},
        {name="delete_cache_file", type="boolean", default=false,
         help="Delete/remove the dbcollection.json cache file.",
         opt = true},
        {name="reset_cache", type="boolean", default=false,
         help="Reset the cache file.",
         opt = true},
        {name="verbose", type="boolean", default=true,
         help="Displays text information (if true).",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    local command = ('import dbcollection as dbc;' ..
                    'dbc.config_cache(field=%s,value=%s,delete_cache=%s, ' ..
                    'delete_cache_dir=%s,delete_cache_file=%s,reset_cache=%s, ' ..
                    'verbose=%s,is_test=%s)')
                    :format(tostring_none(args.field),
                            tostring_none(args.value),
                            tostring_py(args.delete_cache),
                            tostring_py(args.delete_cache_dir),
                            tostring_py(args.delete_cache_file),
                            tostring_py(args.reset_cache),
                            tostring_py(args.verbose),
                            tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end

------------------------------------------------------------------------------------------------------------

function dbcollection.query(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Do simple queries to the cache.

            list all available datasets for download/preprocess.

            Parameters:
            -----------
            pattern : str
                Field name used to search for a matching pattern in cache data.
                (optional, default='info')
            is_test : bool
                Flag used for tests.
                (optional, default=false)

        ]],
        {name="pattern", type="string", default="info",
         help="Field name used to search for a matching pattern in cache data.",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    local command = ('import dbcollection as dbc;' ..
                    'print(dbc.query(pattern=\'%s\',is_test=%s))')
                    :format(args.pattern, tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end

------------------------------------------------------------------------------------------------------------

function dbcollection.info_cache(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Prints the cache contents and other information.

            Parameters
            ----------
            name : str/table, optional
                Name or list of names to be selected for print.
            paths_info : bool, optional
                Print the paths info to screen.
            datasets_info : bool, optional
                Print the datasets info to screen.
            categories_info : bool, optional
                Print the categories keywords info to screen.
            is_test : bool, optional
                Flag used for tests.

        ]],
        {name="name", type="string", default='None',
         help="Name of the dataset to display information.",
         opt = true},
        {name="paths_info", type="boolean", default=true,
         help=" Print the paths info to screen.",
         opt = true},
        {name="datasets_info", type="boolean", default=true,
         help="Print the datasets info to screen.",
         opt = true},
        {name="categories_info", type="boolean", default=true,
         help="Print the paths info to screen.",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true},
    }

    -- parse options
    local args = initcheck(...)

    local command = ('import dbcollection as dbc;' ..
                     'dbc.info_cache(name=%s, paths_info=%s, datasets_info=%s, categories_info=%s, is_test=%s)')
                     :format(tostring_none(args.name),
                             tostring_py(args.paths_info),
                             tostring_py(args.datasets_info),
                             tostring_py(args.categories_info),
                             tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end

------------------------------------------------------------------------------------------------------------

function dbcollection.info_datasets(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Prints information about available and downloaded datasets.

            Parameters
            ----------
            db_pattern : str
                String for matching dataset names available for downloading in the database.
            show_downloaded : bool, optional
                Print the downloaded datasets stored in cache.
            show_available : bool, optional
                Print the available datasets for load/download with dbcollection.

        ]],
        {name="db_pattern", type="string", default='',
         help="String for matching dataset names available for downloading in the database.",
         opt = true},
        {name="show_downloaded", type="boolean", default=true,
         help="Print the downloaded datasets stored in cache.",
         opt = true},
        {name="show_available", type="boolean", default=true,
         help="Print the available datasets for load/download with dbcollection.",
         opt = true},
        {name="is_test", type="boolean", default=false,
         help="Flag used for tests.",
         opt = true},
    }

    -- parse options
    local args = initcheck(...)

    local command = ('import dbcollection as dbc;' ..
                     'dbc.info_datasets(db_pattern=\'%s\', show_downloaded=%s, show_available=%s, is_test=%s)')
                     :format(args.db_pattern,
                             tostring_py(args.show_downloaded),
                             tostring_py(args.show_available),
                             tostring_py(args.is_test))

    os.execute(('python -c "%s"'):format(command))
end
