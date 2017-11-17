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
    local info = fetch_minst_info()
    info.extract_data = true
    info.verbose = true
    info.task = nil
    dbc.download(info)
end

--------------------------------------------------------------------------------
-- Output
--------------------------------------------------------------------------------

return function(_tester_)
    tester = _tester_
    return test
 end
