--[[
    Dataset loader class.
--]]

local hdf5 = require 'hdf5'
local dbcollection = require 'dbcollection.env'
local string_ascii = require 'dbcollection.utils.string_ascii'

local DataLoader = torch.class('dbcollection.DatasetLoader', dbcollection)

------------------------------------------------------------------------------------------------------------

function DataLoader:__init(name, task, data_dir, cache_path)
--[[
    dbcollection's data loading API class.

    Parameters
    ----------
    name : str
        Name of the dataset.
    task : str
        Name of the task.
    data_dir : str
        Path of the dataset's data directory on disk.
    cache_path : str
        Path of the metadata cache file stored on disk.

]]
    assert(name, ('Must input a valid dataset name: %s'):format(name))
    assert(task, ('Must input a valid task name: %s'):format(task))
    assert(data_dir, ('Must input a valid path for the data directory: %s'):format(data_dir))
    assert(cache_path, ('Must input a valid path for the cache file: %s'):format(cache_path))

    -- store information of the dataset
    self.name = name
    self.task = task
    self.data_dir = data_dir
    self.cache_path = cache_path

    -- create a handler for the cache file
    self.file = hdf5.open(self.cache_path, 'r')
    self.root_path = '/default'

    -- make links for all groups (train/val/test/etc) for easier access
    self.sets = {}
    self.object_fields = {}
    local group_default = self.file:read(self.root_path)
    for k, v in pairs(group_default._children) do
        -- add set to the table
        table.insert(self.sets, k)

        -- fetch list of field names that compose the object list.
        local data = self.file:read(('%s/%s/object_fields'):format(self.root_path, k)):all()
        if data:dim()==1 then data = data:view(1,-1) end
        self.object_fields[k] = string_ascii.convert_ascii_to_str(data)
    end
end

------------------------------------------------------------------------------------------------------------

local function assert_value(idx, range_ini, range_end)
    assert(idx)
    assert(range_ini)
    assert(range_end)
    assert(idx>=range_ini and idx<=range_end, string.format('Invalid index value: %d. Valid range: (%d, %d);',
                                                            idx, range_ini, range_end))
end

function DataLoader:get(set_name, field_name, idx)
--[[
    Retrieve data from the dataset's hdf5 metadata file.

    Retrieve the i'th data from the field 'field_name'.

    Parameters
    ----------
    set_name : string
        Name of the set.
    field_name : string
        Name of the data field.
	idx : number/table, optional
        Index number of the field. If the input is a table, it uses it as a range
        of indexes and returns the data for that range.

    Returns
    -------
    torch.Tensor
        Value/list of a field from the metadata cache file.

]]
    assert(set_name, 'Must input a valid set name')
    assert(field_name, 'Must input a valid field name')

    local field_path = ('%s/%s/%s'):format(self.root_path, set_name, field_name)
    local data = self.file:read(field_path)
    local out
    if idx then
        local size = data:dataspaceSize()

        if type(idx) == 'number' then
            assert_value(idx, 1, size[1])
            idx = {idx, idx}
        elseif type(idx) == 'table' then
            assert(next(idx), 'Invalid range. Cannot input an empty table. Valid inputs: nil, {idx} or {idx_ini, idx_end}.')
            assert(#idx>=1 and #idx<=2, 'Invalid range. Must have at most two entries. Valid inputs: nil, {idx} or {idx_ini, idx_end}.')
            if #idx == 1 then
                assert_value(idx[1], 1, size[1])
                idx[2]=idx[1]
            else
                assert_value(idx[1], 1, size[1])
                assert_value(idx[2], 1, size[1])
                assert(idx[2] >= idx[1], 'Invalid range. The first index must be lower or equal to the second one. ' ..
                                         'Valid inputs: nil, {idx} or {idx_ini, idx_end}.')
            end
        else
            error('Invalid index type: %s. Must be either a \'number\' or a \'table\'.')
        end

        local ranges = {idx}
        for i=2, #size do
            table.insert(ranges, {1, size[i]})
        end
        out = data:partial(unpack(ranges))
    else
        out = data:all()
    end

    -- check if the field is 'object_ids'.
    -- If so, add one in order to get the right idx (python uses 0-index)
    if field_name == 'object_ids' then
        return out:add(1)
    else
        return out
    end
end

------------------------------------------------------------------------------------------------------------

function DataLoader:object(set_name, idx, is_value)
--[[
    Retrieves a list of all fields' indexes/values of an object composition.

    Retrieves the data's ids or contents of all fields of an object.

    It works by calling :get() for each field individually and grouping
    them into a list.

    Parameters
    ----------
    set_name : str
        Name of the set.
    idx : int, long, list
        Index number of the field. If it is a list, returns the data
        for all the value indexes of that list
    is_value : bool, optional
       Outputs a tensor of indexes (if false)
       or a table of tensors/values (if true).

    Returns:
    --------
    table
        Returns a table of indexes (or values, i.e. tensors, if is_value=True).

]]
    assert(set_name, 'Must input a valid set name')
    local is_value = is_value or false

    local set_path = ('%s/%s/'):format(self.root_path,set_name)

    local indexes = self:get(set_name, 'object_ids', idx)

    if is_value then
        local out = {}
        for i=1, indexes:size(1) do
            local data = {}
            for k, field in ipairs(self.object_fields[set_name]) do
                if indexes[i][k] > 0 then
                    table.insert(data, self:get(set_name, field, indexes[i][k]))
                else
                    table.insert(data, {})
                end
            end
            table.insert(out, data)
        end
        if #out > 1 then
            return out
        else
            return out[1]
        end
    else
        return indexes
    end
end

------------------------------------------------------------------------------------------------------------

function DataLoader:size(set_name, field_name)
--[[
    Size of a field.

    Returns the number of the elements of a field_name.

    Parameters
    ----------
    set_name : str
        Name of the set.
    field_name : str, optional
        Name of the data field.

    Returns:
    --------
    table
        Returns the the size of the object list.

]]
    assert(set_name, ('Must input a valid set name: %s'):format(set_name))

    local field_name = field_name or 'object_ids'
    local field_path = ('%s/%s/%s'):format(self.root_path, set_name, field_name)
    local data = self.file:read(field_path)
    return data:dataspaceSize()
end

------------------------------------------------------------------------------------------------------------

function DataLoader:list(set_name)
--[[
    Lists all fields' names.

    Parameters
    ----------
    set_name : str
        Name of the set.

    Returns
    -------
    table
        List of all data fields names of the dataset.

]]
    assert(set_name, ('Must input a valid set name: %s'):format(set_name))

    local set_path = ('%s/%s'):format(self.root_path, set_name)
    local data = self.file:read(set_path)
    local list_fields = {}
    for k, _ in pairs(data._children) do
        table.insert(list_fields, k)
    end
    return list_fields
end

------------------------------------------------------------------------------------------------------------

function DataLoader:object_field_id(set_name, field_name)
--[[
    Retrieves the index position of a field in the 'object_ids' list.

    Parameters
    ----------
    set_name : str
        Name of the set.
    field_name : str
        Name of the data field.

    Returns
    -------
    number
        Index of the field_name on the list.

    Raises
    ------
    error
        If field_name does not exist on the 'object_fields' list.
]]
    assert(set_name, ('Must input a valid set name: %s'):format(set_name))
    assert(field_name, ('Must input a valid field_name name: %s'):format(field_name))

    for k, field in pairs(self.object_fields[set_name]) do
        if string.match(field_name, field) then
            return k
        end
    end
    error(('Field name \'%s\' does not exist.'):format(field_name))
end

------------------------------------------------------------------------------------------------------------

--[[ Get the maximum length of all elements (strings only). ]]
local function get_max_size(tableA, key)
    local max = 0
    for k=1, #tableA do
        max = math.max(max, #tableA[k][key])
    end
    return max
end

local function string_pad(str, len, char)
    if char == nil then char = ' ' end
    return str .. string.rep(char, len - #str)
end

local function is_in_table(tableA, value)
    assert(tableA)
    assert(value)
    for k, v in pairs(tableA) do
        if v == value then
            return true
        end
    end
    return false
end


--[[ Split the field names into two separate lists for display. ]]--
function DataLoader:_get_fields_lists_info(set_name)
    assert(set_name)

    -- get all field names
    local field_names = self:list(set_name)
    table.sort(field_names)  --sort the list alphabetically

    -- split fields names into two tables
    local fields_info, list_info = {}, {}
    for k=1, #field_names do
        local field_name = field_names[k]
        local f = self.file:read(self.root_path .. '/' .. set_name .. '/' .. field_name)
        local size = f:dataspaceSize()
        local ranges = {1}
        for i=2, #size do
            table.insert(ranges, {1, size[i]})
        end
        local tensor = f:partial(unpack(ranges))
        local s_shape = ''
        for i=1, #size do
            s_shape = s_shape .. tostring(size[i])
            if i < #size then
                s_shape = s_shape .. ', '
            end
        end
        s_shape = '{' .. s_shape .. '}'
        local s_type = tensor:type()
        if field_name:find('list_') then
            table.insert(list_info, {name = field_name,
                                     shape = 'shape = ' .. s_shape,
                                     type = 'dtype = ' .. s_type})
        else
            local s_obj = ''
            if is_in_table(self.object_fields[set_name], field_name) then-- pcall(self:object_field_id, set_name, field_name) then
                s_obj = ("(in 'object_ids', position = %d)"):format(self:object_field_id(set_name, field_name))
            end
            table.insert(fields_info, {name = field_name,
                                       shape = 'shape = ' .. s_shape,
                                       type = 'dtype = ' .. s_type,
                                       obj = s_obj})
        end
    end
    return fields_info, list_info
end


--[[ Prints information about the data fields of a set. ]]--
function DataLoader:_print_info(set_name)
--[[
    Prints information about the data fields of a set.

    Displays information of all fields available like field name,
    size and shape of all sets. If a 'set_name' is provided, it
    displays only the information for that specific set.

    This method provides the necessary information about a data set
    internals to help determine how to use/handle a specific field.

    Parameters
    ----------
    set_name : str
        Name of the set.

]]
    assert(set_name)

    -- get a list of field names and a list of list fields
    local fields_info, list_info = self:_get_fields_lists_info(set_name)

    print(('\n> Set: %s'):format(set_name))

    -- prints all fields except list_*
    local maxsize_name = get_max_size(fields_info, 'name') + 8
    local maxsize_shape = get_max_size(fields_info, 'shape') + 3
    local maxsize_type = get_max_size(fields_info, 'type') + 3
    for k=1, #fields_info do
        local s_name = string_pad('   - ' .. fields_info[k].name .. ',', maxsize_name, ' ')
        local s_shape = string_pad(fields_info[k].shape .. ',', maxsize_shape, ' ')
        local s_obj = fields_info[k].obj
        local s_type
        if #s_obj > 1 then
            s_type = string_pad(fields_info[k].type .. ',', maxsize_type, ' ')
        else
            s_type = string_pad(fields_info[k].type, maxsize_type, ' ')
        end
        print(string.format('%s %s %s %s', s_name, s_shape, s_type, s_obj))
    end

    -- prints only list fields
    if next(list_info) then
        print('\n   (Pre-ordered lists)')
        local maxsize_name = get_max_size(list_info, 'name') + 8
        local maxsize_shape = get_max_size(list_info, 'shape') + 3
        for k=1, #list_info do
            local s_name = string_pad('   - ' .. list_info[k].name .. ',', maxsize_name, ' ')
            local s_shape = string_pad(list_info[k].shape .. ',', maxsize_shape, ' ')
            local s_type  = list_info[k].type
            print(string.format('%s %s %s', s_name, s_shape, s_type))
        end
    end
end


function DataLoader:info(set_name)
--[[
    Prints information about the data fields of a set.

    Displays information of all fields available like field name,
    size and shape of all sets. If a 'set_name' is provided, it
    displays only the information for that specific set.

    This method provides the necessary information about a data set
    internals to help determine how to use/handle a specific field.

    Parameters
    ----------
    set_name : str
        Name of the set.
]]
    if set_name then
        self:_print_info(set_name)
    else
        for _, set_name in pairs(self.sets) do
            self:_print_info(set_name)
        end
    end
end