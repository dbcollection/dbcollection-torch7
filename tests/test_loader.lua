--[[
    loader.lua unit tests.

    Warning: Requires Torch7 to be installed.
--]]


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

local function generate_dataset()
    local sets = {
        train = 10,
        test = 5
    }

    local dataset = {}
    for set, size in pairs(sets) do
        local field_name = 'data'
        dataset[set] = {}
        dataset[set][field_name] = torch.repeatTensor(torch.range(1,10), size, 1)
        dataset[set]['object_fields'] = str_to_ascii({field_name})
        dataset[set]['object_ids'] = torch.range(1, size)
    end

    return dataset
end

local function create_hdf5_file()
    if paths.filep(hdf5_file) then
        os.execute('rm -rf ' .. hdf5_file)
    end
    local dirname = paths.dirname(hdf5_file)
    if not paths.dirp(dirname) then
        os.execute('mkdir -p ' .. dirname)
    end
    return hdf5.open(hdf5_file, 'w')
end

local function populate_hdf5_file(h5_obj, dataset)
    for set, data in pairs(dataset) do
        for field, val in pairs(data) do
            h5_obj:write('/' .. set .. '/' .. field, val)
        end
    end
    h5_obj:close()
end

local function create_dummy_hdf5_file()
    local h5_obj = create_hdf5_file()
    local dataset = generate_dataset()
    populate_hdf5_file(h5_obj, dataset)
end

create_dummy_hdf5_file()

local function load_dummy_hdf5_file()
    if not paths.filep(hdf5_file) then
        create_dummy_hdf5_file()
    end
    return hdf5.open(hdf5_file, 'r')
end

local function load_dummy_hdf5_file_FieldLoader(path)
    assert(path)
    local h5obj = load_dummy_hdf5_file()
    local obj_id = 1
    local field_loader = dbc.FieldLoader(h5obj:read(path), obj_id)
    return field_loader, dataset
end

local function load_test_data_FieldLoader(set)
    local set = set or 'train'
    local path = '/' .. set .. '/data'
    local field_loader = load_dummy_hdf5_file_FieldLoader(path)

    local dataset = generate_dataset()
    local set_data = dataset[set]

    return field_loader, set_data
end


--------------------------------------------------------------------------------
-- Tests
--------------------------------------------------------------------------------

function test.test_FieldLoader__init()
    local h5obj = load_dummy_hdf5_file()

    local obj_id = 1
    local field_loader = dbc.FieldLoader(h5obj:read('/train/data'), obj_id)

    tester:assert(field_loader._in_memory == false)
    tester:eq(field_loader.name, 'data', 'Names are note the same')
end

function test.test_FieldLoader_get_single_obj()
    local field_loader, set_data = load_test_data_FieldLoader('train')

    local id = 1
    local data = field_loader:get(id)

    tester:eq(data, set_data['data'][1]:view(1, -1))
end

function test.test_FieldLoader_get_all_obj()
    local field_loader, set_data = load_test_data_FieldLoader('train')

    local data = field_loader:get()

    tester:eq(data, set_data['data'])
end

function test.test_FieldLoader_size()
    local field_loader, set_data = load_test_data_FieldLoader('train')

    local size = field_loader:size()

    tester:eq(size, set_data['data']:size():totable(), 'Sizes are not the same')
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
