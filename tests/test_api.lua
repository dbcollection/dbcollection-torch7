--[[
    api.lua unit tests.

    Warning: Requires Torch7 to be installed.
--]]

-- initializations
require 'paths'
local dbc = require 'dbcollection.env'

local tester
local test = torch.TestSuite()


--------------------------------------------------------------------------------
-- Data setup
--------------------------------------------------------------------------------

local data_dir= paths.concat(paths.home, 'tmp', 'download_data')

function clean_data()
    -- datasets
    local datasets = {
        cifar10 = paths.concat(data_dir, 'cifar10'),
        mnist = paths.concat(data_dir, 'mnist'),
    }

    -- remove data files for all datasets
    for db_name, db_path in pairs(datasets) do
        if paths.filep(db_path) then
            paths.rmdir(db_path)
        end
    end
end

if os.getenv('travis_dbcollection_test') then
    print('==> Cleaning test data files...')
    clean_data()
    print('==> Done!')
end

function fetch_minst_info(db_name)
    return {
        name = db_name or 'mnist',
        task = 'classification',
        data_dir = data_dir,
        verbose = false,
        is_test = true,
    }
end


--------------------------------------------------------------------------------
-- Tests
--------------------------------------------------------------------------------


function test.test_download_mnist()
    local info = fetch_minst_info()
    info.extract_data = true
    info.verbose = true
    info.task = nil
    dbc.download(info)
end

function test.test_download_cifar10()
    local info = fetch_minst_info('cifar10')
    info.extract_data = true
    info.verbose = true
    info.task = nil
    dbc.download(info)
end

function test.test_process_mnist()
    local info = fetch_minst_info()
    info.data_dir = nil
    dbc.process(info)
end

--function test.test_load_mnist()
--    local info = fetch_minst_info()
--    info.is_test = true
--    local db = dbc.load(info)
--    tester:eq(db.db_name, info.name)
--    tester:eq(db.task, info.task)
--    tester:eq(db.data_dir, paths.concat(info.data_dir, info.name))
--end

function test.test_add()
    dbc.add({name='new_db',
             task='new_task',
             data_dir='new/path/db',
             file_path='newdb.h5',
             keywords={'new_category'},
             is_test=true})
end

function test.test_add2()
    for i=1, 2 do
        dbc.add({name='new_db',
                 task='new_task',
                 data_dir='new/path/db',
                 file_path='newdb.h5',
                 keywords={'new_category'}, is_test=true})
    end
end

function test.test_remove()
    dbc.add({name='new_db', task='new_task', data_dir='new/path/db', file_path='newdb.h5', keywords={'new_category'}, is_test=true})
    dbc.remove({name='new_db', task='new_task', delete_data=true, is_test=true})
end

--------------------------------------------------------------------------------
-- Output
--------------------------------------------------------------------------------

return function(_tester_)
    tester = _tester_
    return test
 end
