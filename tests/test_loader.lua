--[[
    loader.lua unit tests.

    Warning: Requires Torch7 to be installed.
--]]

-- initializations
require 'paths'
local dbc = require 'dbcollection.env'
local hdf5 = require 'hdf5'

local str_to_ascii = dbc.utils.string_ascii.convert_str_to_ascii

local tester
local test = torch.TestSuite()


--------------------------------------------------------------------------------
-- Data setup
--------------------------------------------------------------------------------

local hdf5_file = paths.concat(paths.home, 'tmp', 'dbcollection', 'dummy.h5')

local function create_dummy_hdf5_file()
    if paths.filep(hdf5_file) then
        os.execute('rm -rf ' .. hdf5_file)
    end
    local dirname = paths.dirname(hdf5_file)
    if not paths.dirp(dirname) then
        os.execute('mkdir -p ' .. dirname)
    end
    local h5_obj = hdf5.open(hdf5_file, 'w')
    local object_fields = str_to_ascii({'data'})
    h5_obj:write('/train/data', torch.repeatTensor(torch.range(1,10), 10, 1))
    h5_obj:write('/train/object_fields', object_fields)
    h5_obj:write('/train/object_ids', torch.range(1,10))
    h5_obj:write('/test/data', torch.repeatTensor(torch.range(1,10), 5, 1))
    h5_obj:write('/test/object_fields', object_fields)
    h5_obj:write('/test/object_ids', torch.range(1,5))
    h5_obj:close()
end

create_dummy_hdf5_file()

local function load_dummy_hdf5_file()
    if not paths.filep(hdf5_file) then
        create_dummy_hdf5_file()
    end
    return hdf5.open(hdf5_file, 'r')
end


function setUp()
    local home_dir = paths.home
    local name = 'mnist'
    local task = 'classification'

    local data_dir = paths.concat(home_dir, 'tmp', 'dbcollection', 'mnist', 'data')
    local cache_path = paths.concat(home_dir, 'tmp', 'dbcollection', 'mnist', 'classification.h5')

    --if not paths.filep(cache_path) then
    --    local db = dbc.load({name=name, task=task, is_test=true})
    --
    --    ----------------------------------------
    --    -- temporary fix.
    --    -- To be removed when the new dbcollection
    --    -- Python module version is uploaded to pip.
    --    if not paths.filep(cache_path) then
    --        cache_path = paths.concat(home_dir, 'tmp', 'dbcollection', 'mnist', 'detection.h5')
    --    end
    --    ----------------------------------------
    --end

    -- initialize object
    local loader = dbc.DataLoader(name, task, data_dir, cache_path)

    local utils = dbc.utils

    return loader, utils
end


--------------------------------------------------------------------------------
-- Tests
--------------------------------------------------------------------------------

function test.test_split_string()
    local str = '/train/data'
    local split = str:split('/')
    tester:assert(#split == 2)
end

function test.test_FieldLoader__init()
    local h5obj = load_dummy_hdf5_file()
    local obj_id = 1
    local fieldLoader = dbc.FieldLoader(h5obj:read('/train/data'), obj_id)
    tester:assert(fieldLoader ~= nil)
    tester:assert(fieldLoader._in_memory ~= nil)
    tester:eq(fieldLoader.name, 'data', 'Names are note the same')
    tester:eq(fieldLoader.size, {10, 10}, 'Sizes are not the same')
end

function test.test_FieldLoader_get()
end

function test.test_FieldLoader_size()
end

function test.test_FieldLoader_object_field_id()
end

function test.test_FieldLoader_info()
end

function test.test_FieldLoader_to_memory()
end

function test.test_FieldLoader__len__()
end

function test.test_FieldLoader__tostring__()
end

function test.test_FieldLoader__index__()
end

------------------------------------------------------------------------------------------------------------

function test.test_SetLoader__init()
    local h5obj = load_dummy_hdf5_file()
    local setLoader = dbc.SetLoader(h5obj:read('/test'))
    tester:assert(setLoader ~= nil)
    tester:eq(setLoader.set, 'test')
    tester:eq(setLoader._object_fields, 'data')
    tester:eq(setLoader.nelems, 5)
end

function test.test_SetLoader_get()
end

function test.test_SetLoader_object()
end

function test.test_SetLoader_size()
end

function test.test_SetLoader_list()
end

function test.test_SetLoader_info()
end

function test.test_SetLoader__len__()
end

function test.test_SetLoader__tostring__()
end

------------------------------------------------------------------------------------------------------------

function test.test_DataLoader_init()
    local name = 'some_db'
    local task = 'task'
    local data_dir = './some/dir'
    local file = hdf5_file
    local DataLoader = dbc.DataLoader(name, task, data_dir, file)
    tester:assert(DataLoader ~=  nil)
    tester:eq(DataLoader.db_name, name)
    tester:eq(DataLoader.task, task)
    tester:eq(DataLoader.data_dir, data_dir)
    tester:eq(DataLoader.hdf5_filepath, file)
    tester:eq(DataLoader.sets, {'test','train'})
end

function test.test_DataLoader_open_hdf5_file()
    --local loader, utils  = setUp()
    --tester:assert(loader:_open_hdf5_file() ~=  nil)
end

function test.test_DataLoader_get_set_names()
    --local loader, utils  = setUp()
    --tester:eq(loader.sets, loader:_get_set_names())
end

function test.test_DataLoader_get_object_fields()
    --local loader, utils  = setUp()
    --tester:eq(loader.object_fields, loader:_get_object_fields())
end

function test.test_DataLoader_get()
    --local loader, utils  = setUp()
end

function test.test_DataLoader_object()
    --local loader, utils  = setUp()
end

function test.test_DataLoader_size()
    --local loader, utils  = setUp()
end

function test.test_DataLoader_list()
    --local loader, utils  = setUp()
end

function test.test_DataLoader_object_field_id()
    --local loader, utils  = setUp()
end

function test.test_DataLoader_info()
    --local loader, utils  = setUp()
end

function test.test_DataLoader__len__()
    --local loader, utils  = setUp()
end

function test.test_DataLoader__tostring__()
    --local loader, utils  = setUp()
end


--------------------------------------------------------------------------------
-- Output
--------------------------------------------------------------------------------

return function(_tester_)
    tester = _tester_
    return test
end
