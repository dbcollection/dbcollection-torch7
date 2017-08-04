local __main__ = package.loaded['dbcollection.env'] == nil

local dbc = require 'dbcollection.env'

if __main__ then
   require 'dbcollection'
end

local tester = torch.Tester()
--tester:add(paths.dofile('test_loader.lua')(tester))
--tester:add(paths.dofile('test_manager.lua')(tester))
tester:add(paths.dofile('test_string_ascii.lua')(tester))

function dbc.test(tests)
   tester:run(tests)
   return tester
end

if __main__ then
   require 'dbcollection'
   if #arg > 0 then
      dbc.test(arg)
   else
      dbc.test()
   end
end