--[[
    loader.lua unit tests.

    Warning: Requires Torch7 to be installed.
--]]


-- initializations
require 'paths'
local dbc = require 'dbcollection.env'

local tester
local test = torch.TestSuite()


------------
-- Tests
------------

function setUp()
    local home_dir = paths.home
    local name = 'mnist'
    local task = 'classification'

    local data_dir = paths.concat(home_dir, 'dbcollection', 'mnist', 'data')
    local cache_path = paths.concat(home_dir, 'dbcollection', 'mnist', 'classification.h5')

    -- initialize object
    local loader = dbc.DatasetLoader(name, task, data_dir, cache_path)

    local utils = dbc.utils

    return loader, utils
end

function test.test_get_all()
    local sample_classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    -- setup data loader
    local loader, utils = setUp()

    -- call get() method
    local data = loader:get('train', 'classes');

    -- convert double to char
    local classes = utils.string_ascii.convert_ascii_to_str(data)

    tester:eq(sample_classes, classes)
end

function test.test_get_range()
    local sample_classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    -- setup data loader
    local loader, utils = setUp()

    -- call get() method
    local data = loader:get('train', 'classes', {1,10})

    -- convert double to char
    local classes = utils.string_ascii.convert_ascii_to_str(data)

    tester:eq(sample_classes, classes)
end

function test.test_get_range2()
    local sample_classes = {'5', '6', '7', '8', '9'}

    -- setup data loader
    local loader, utils = setUp()

    -- call get() method
    local data = loader:get('train', 'classes', {6,10})

    -- convert double to char
    local classes = utils.string_ascii.convert_ascii_to_str(data)

    tester:eq(sample_classes, classes)
end

function test.test_object_single()
    local sample_ids = torch.Tensor({{1, 6}})

    -- setup data loader
    local loader = setUp()

    -- call object() method
    local ids = loader:object('train', 1)

    tester:eq(sample_ids:double(), ids:double())
end

function test.test_object_two()
    local sample_ids = torch.Tensor({{1, 6}, {2, 1}})

    -- setup data loader
    local loader = setUp()

    -- call object() method
    local ids = loader:object('train', {1, 2})

    tester:eq(sample_ids:double(), ids:double())
end

function test.test_size_1()
    local sample_ids = {10, 2}

    -- setup data loader
    local loader = setUp()

    -- call object() method
    local cls_size = loader:size('train', 'classes')

    tester:eq(sample_ids, cls_size)
end

function test.test_size_2()
    local sample_ids = {60000, 2}

    -- setup data loader
    local loader = setUp()

    -- call object() method
    local cls_size = loader:size('train')

    tester:eq(sample_ids, cls_size)
end

function test.test_list()
    local sample_field_names = {'classes',
                                'images',
                                'labels',
                                'list_images_per_class',
                                'object_fields',
                                'object_ids'}

    -- setup data loader
    local loader = setUp()

    -- call object() method
    local field_names = loader:list('train')
    table.sort(field_names)

    tester:eq(sample_field_names, field_names)
end

function test.test_object_field_id_1()
    local sample_idx = 1

    -- setup data loader
    local loader = setUp()

    -- call object() method
    local res = loader:object_field_id('train', 'images')

    tester:eq(sample_idx, res)
end

function test.test_object_field_id_2()
    local sample_idx = 2

    -- setup data loader
    local loader = setUp()

    -- call object() method
    local res = loader:object_field_id('train', 'labels')

    tester:eq(sample_idx, res)
end

function test.test_info()
    -- setup data loader
    local loader = setUp()

    -- call object() method
    loader:info()

    tester:assert(true)
end

function test.test_info_set()
    -- setup data loader
    local loader = setUp()

    -- call object() method
    loader:info('test')

    tester:assert(true)
end


return function(_tester_)
   tester = _tester_
   return test
end
