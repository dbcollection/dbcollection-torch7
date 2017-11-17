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
-- Tests
--------------------------------------------------------------------------------


function fetch_minst_info()
    return {
        name = 'mnist',
        task = 'classification',
        data_dir = paths.concat(paths.home, 'tmp', 'download_data'),
        verbose = false,
        is_test = true,
    }
end


--------------------------------------------------------------------------------
-- Output
--------------------------------------------------------------------------------

return function(_tester_)
    tester = _tester_
    return test
 end
 