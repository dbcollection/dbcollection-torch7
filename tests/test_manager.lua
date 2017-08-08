--[[
    manager.lua unit tests.

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

function fetch_minst_info()
    return {
        name = 'mnist',
        task = 'classification',
        data_dir = paths.concat(paths.home, 'tmp', 'download_data'),
        verbose = false,
        is_test = true,
    }
end

function test.test_download()
    local info = fetch_minst_info()
    info.extract_data = true
    info.verbose = true
    info.task = nil
    dbc.download(info)
end

function test.test_process()
    local info = fetch_minst_info()
    info.data_dir = nil
    dbc.process(info)
end

function test.test_load()
    local info = fetch_minst_info()
    info.is_test = true
    local db = dbc.load(info)
    tester:eq(db.name, info.name)
    tester:eq(db.task, info.task)
    tester:eq(db.data_dir, paths.concat(info.data_dir, info.name))
end

function test.test_add()
    dbc.add({name='new_db', task='new_task', data_dir='new/path/db', file_path='newdb.h5', keywords={'new_category'}, is_test=true})
end

function test.test_add2()
    dbc.add({name='new_db', task='new_task', data_dir='new/path/db', file_path='newdb.h5', keywords={'new_category'}, is_test=true})
    dbc.add({name='new_db', task='new_task', data_dir='new/path/db', file_path='newdb.h5', keywords={'new_category'}, is_test=true})
end

function test.test_remove()
    dbc.add({name='new_db', task='new_task', data_dir='new/path/db', file_path='newdb.h5', keywords={'new_category'}, is_test=true})
    dbc.remove({name='new_db', task='new_task', delete_data=true, is_test=true})
end

function test.test_config_cache()
    dbc.config_cache({reset_cache=true, is_test=true})
end

function test.test_query()
    dbc.query('info', true)
end

function test_info()
    dbc.info({is_test=True})
end

function test.test_info_list_datasets()
    dbc.info({name='all', is_test=true})
end


return function(_tester_)
   tester = _tester_
   return test
end
