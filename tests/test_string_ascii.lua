--[[
    string_ascii.lua unit tests.

    Warning: Requires Torch7 to be installed.
--]]


-- initializations
local ffi = require 'ffi'
local dbc = require 'dbcollection.env'

local tester
local test = torch.TestSuite()


function test.test_convert_str_to_ascii__string()
    local toascii_ = dbc.utils.string_ascii.convert_str_to_ascii
    local str = 'test_string'
    local str_tensor = torch.CharTensor(1,#str+1):fill(0)
    ffi.copy(str_tensor[1]:data(), str)
    tester:eq(str_tensor, toascii_(str))
end

function test.test_convert_str_to_ascii__table()
    local toascii_ = dbc.utils.string_ascii.convert_str_to_ascii
    local str = {'test_string1', 'test_string2', 'test_string3'}
    local max_length = #str[1] + 1
    local str_tensor = torch.CharTensor(#str, max_length):fill(0)
    local s_data = str_tensor:data()
    for i=1, #str do
        ffi.copy(s_data, str[i])
        s_data = s_data + max_length
    end
    tester:eq(str_tensor, toascii_(str))
end

function test.test_convert_ascii_to_str__CharTensor_1D()
    local tostring_ = dbc.utils.string_ascii.convert_ascii_to_str
    local str = 'test_string'
    local str_tensor = torch.CharTensor(1,#str+1):fill(0)
    ffi.copy(str_tensor:data(), str)
    tester:eq(str, tostring_(str_tensor))
end

function test.test_convert_ascii_to_str__CharTensor_2D()
    local tostring_ = dbc.utils.string_ascii.convert_ascii_to_str
    local str = {'test_string1', 'test_string2', 'test_string3'}
    local max_length = #str[1] + 1
    local str_tensor = torch.CharTensor(#str, max_length):fill(0)
    local s_data = str_tensor:data()
    for i=1, #str do
        ffi.copy(s_data, str[i])
        s_data = s_data + max_length
    end
    tester:eq(str, tostring_(str_tensor))
end

function test.test_convert_ascii_to_str__ByteTensor_2D()
    local tostring_ = dbc.utils.string_ascii.convert_ascii_to_str
    local str = {'test_string1', 'test_string2', 'test_string3'}
    local max_length = #str[1] + 1
    local str_tensor = torch.CharTensor(#str, max_length):fill(0)
    local s_data = str_tensor:data()
    for i=1, #str do
        ffi.copy(s_data, str[i])
        s_data = s_data + max_length
    end
    tester:eq(str, tostring_(str_tensor:byte()))
end

return function(_tester_)
   tester = _tester_
   return test
end