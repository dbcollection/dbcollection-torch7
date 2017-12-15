--[[
    Dataset's metadata loader classes.
--]]

local hdf5 = require 'hdf5'
local dbcollection = require 'dbcollection.env'
local string_ascii = require 'dbcollection.utils.string_ascii'

local DataLoader = torch.class('dbcollection.DataLoader', dbcollection)
local SetLoader = torch.class('dbcollection.SetLoader', dbcollection)
local FieldLoader = torch.class('dbcollection.FieldLoader', dbcollection)

---------------------------------------------------------------------------------------------------

local function get_value_id_in_list(val, list)
    for i=1, #list do
        if list[i] == val then
            return i
        end
    end
    return {}
end

local function is_val_in_table(value, source)
    for key, val in pairs(source) do
        if val == value then
            return true
        end
    end
    return false
end

local function concat_shape_string(source, new_string, is_not_last)
    local output = source .. new_string
    if is_not_last then
        output = output .. ', '
    end
    return output
end

local function get_data_shape(size)
    local shape="("
    for j=1, #size do
        shape = concat_shape_string(shape, size[j], j < #size)
    end
    shape = shape .. ')'
    return shape
end

local function get_atomic_indexes(dim)
    local idx = {}
    for i=1, dim do
        table.insert(idx, {1,1})
    end
    return idx
end

local function get_data_type_hdf5(hdf5_dataset, size)
    local idx = get_atomic_indexes(#size)
    local data_sample = hdf5_dataset:partial(unpack(idx))
    return torch.type(data_sample)
end

---------------------------------------------------------------------------------------------------

function DataLoader:__init(name, task, data_dir, hdf5_filepath)
--[[
    Dataset metadata loader class.

    This class contains several methods to fetch data from a hdf5 file
    by using simple, easy to use functions for (meta)data handling.

    Parameters
    ----------
    name : str
        Name of the dataset.
    task : str
        Name of the task.
    data_dir : str
        Path of the dataset's data directory on disk.
    hdf5_filepath : str
        Path of the metadata cache file stored on disk.

    Attributes
    ----------
    db_name : str
        Name of the dataset.
    task : str
        Name of the task.
    data_dir : str
        Path of the dataset's data directory on disk.
    hdf5_filepath : str
        Path of the hdf5 metadata file stored on disk.
    hdf5_file : h5py._hl.files.File
        hdf5 file object handler.
    root_path : str
        Default data group of the hdf5 file.
    sets : tuple
        List of names of set splits (e.g. train, test, val, etc.)
    object_fields : dict
        Data field names for each set split.
]]
    assert(name, ('Must input a valid dataset name: %s'):format(name))
    assert(task, ('Must input a valid task name: %s'):format(task))
    assert(data_dir, ('Must input a valid path for the data directory: %s'):format(data_dir))
    assert(hdf5_filepath, ('Must input a valid path for the cache file: %s'):format(cache_path))

    -- store information of the dataset
    self.db_name = name
    self.task = task
    self.data_dir = data_dir
    self.hdf5_filepath = hdf5_filepath

    -- create a handler for the cache file
    self.file = self:_open_hdf5_file()
    self.root_path = '/'

    self.sets = self:_get_set_names()
    self.object_fields = self:_get_object_fields()

    -- make links for all groups (train/val/test/etc) for easier access
    self:_set_SetLoaders()
end

function DataLoader:_open_hdf5_file()
    return hdf5.open(self.hdf5_filepath, 'r')
end

function DataLoader:_get_set_names()
    local sets = {}
    local group_default = self.file:read(self.root_path)
    for k, v in pairs(group_default._children) do
        table.insert(sets, k)
    end
    return sets
end

function DataLoader:_get_object_fields()
    local object_fields = {}
    for _, set in pairs(self.sets) do
        object_fields[set] = self:_get_object_fields_data_from_set(set)
    end
    return object_fields
end

function DataLoader:_get_object_fields_data_from_set(set)
    local hdf5_dataset_path = self.root_path .. set ..'/object_fields'
    local object_fields_data = self.file:read(hdf5_dataset_path):all()
    if object_fields_data:dim() == 1 then
        object_fields_data = object_fields_data:view(1,-1)
    end
    return string_ascii.convert_ascii_to_str(object_fields_data)
end

function DataLoader:_set_SetLoaders()
    for _, set in pairs(self.sets) do
        local hdf5_group_path = self.root_path .. set
        self[set] = dbcollection.SetLoader(self:_get_hdf5_group(hdf5_group_path), set)
    end
end

function DataLoader:_get_hdf5_group(path)
    return self.file:read(path)
end

function DataLoader:get(set_name, field, idx)
--[[
    Retrieves data from the dataset's hdf5 metadata file.

    This method retrieves the i'th data from the hdf5 file with the
    same 'field' name. Also, it is possible to retrieve multiple values
    by inserting a list/tuple of number values as indexes.

    Parameters
    ----------
    set_name : str
        Name of the set.
    field : str
        Field name.
    idx : int/list/tuple, optional
        Index number of the field. If it is a list, returns the data
        for all the value indexes of that list.

    Returns
    -------
    np.ndarray
        Numpy array containing the field's data.
    list
        List of numpy arrays if using a list of indexes.
]]
    assert(set_name, ('Must input a valid set name: %s'):format(set_name))
    assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                :format(set_name))
    assert(field, ('Must input a valid field name: %s'):format(field))
    return self[set_name]:get(field, idx)
end

function DataLoader:object(set_name, idx, convert_to_value)
--[[
    Retrieves a list of all fields' indexes/values of an object composition.

    Retrieves the data's ids or contents of all fields of an object.

    It basically works as calling the get() method for each individual field
    and then groups all values into a list w.r.t. the corresponding order of
    the fields.

    Parameters
    ----------
    set_name : str
        Name of the set.
    idx : int/list/tuple, optional
        Index number of the field. If it is a list, returns the data
        for all the value indexes of that list. If no index is used,
        it returns the entire data field array.
    convert_to_value : bool, optional
        If False, outputs a list of indexes. If True,
        it outputs a list of arrays/values instead of indexes.

    Returns
    -------
    list
        Returns a list of indexes or, if convert_to_value is True,
        a list of data arrays/values.
]]
    assert(set_name, ('Must input a valid set name: %s'):format(set_name))
    assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                :format(set_name))
    return self[set_name]:object(idx, convert_to_value or false)
end

function DataLoader:size(set_name, field)
--[[
    Size of a field.

    Returns the number of the elements of a field.

    Parameters
    ----------
    set_name : str, optional
        Name of the set.
    field : str, optional
        Name of the field in the metadata file.

    Returns
    -------
    list
        Returns the size of a field.
]]
    local field = field or 'object_ids'
    if set_name then
        self:_get_set_size(set_name, field)
    else
        return self:_get_set_size_all(field)
    end
end

function DataLoader:_get_set_size(set_name, field)
    assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                :format(set_name))
    assert(field, 'Must input a field')
    return self[set_name]:size(field)
end

function DataLoader:_get_set_size_all(field)
    assert(field, 'Must input a field')
    local out = {}
    for set_name, _ in pairs(self.sets) do
        out[set_name] = self:_get_set_size(set_name, field)
    end
    return out
end

function DataLoader:list(set_name)
--[[
    List of all field names of a set.

    Parameters
    ----------
    set_name : str, optional
        Name of the set.

    Returns
    -------
    list
        List of all data fields of the dataset.
]]
    if set_name then
        return self:_get_set_list(set_name)
    else
        return self:_get_set_list_all()
    end
end

function DataLoader:_get_set_list(set_name)
    assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                :format(set_name))
    return self[set_name]:list()
end

function DataLoader:_get_set_list_all()
    local out = {}
    for set_name, _ in pairs(self.sets) do
        out[set_name] = self[set_name]:list()
    end
    return out
end

function DataLoader:object_field_id(set_name, field)
--[[
    Retrieves the index position of a field in the 'object_ids' list.

    This method returns the position of a field in the 'object_ids' object.
    If the field is not contained in this object, it returns a null value.

    Parameters
    ----------
    set_name : str
        Name of the set.
    field : str
        Name of the field in the metadata file.

    Returns
    -------
    int
        Index of the field in the 'object_ids' list.
]]
    assert(set_name, ('Must input a valid set name: %s'):format(set_name))
    assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                :format(set_name))
    assert(field, ('Must input a valid field name: %s'):format(field))
    return self[set_name]:object_field_id(field)
end

function DataLoader:info(set_name)
--[[
    Prints information about all data fields of a set.

    Displays information of all fields of a set group inside the hdf5
    metadata file. This information contains the name of the field, as well
    as the size/shape of the data, the data type and if the field is
    contained in the 'object_ids' list.

    If no 'set_name' is provided, it displays information for all available
    sets.

    This method only shows the most useful information about a set/fields
    internals, which should be enough for most users in helping to
    determine how to use/handle a specific dataset with little effort.

    Parameters
    ----------
    set_name : str, optional
        Name of the set.
]]
    if set_name then
        self:_get_set_info(set_name)
    else
        self:_get_set_info_all()
    end
end

function DataLoader:_get_set_info(set_name)
    assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                :format(set_name))
    self[set_name]:info()
end

function DataLoader:_get_set_info_all()
    for set_name, _ in pairs(self.sets) do
        self[set_name]:info()
    end
end

function DataLoader:__len__()
    return #self.sets
end

function DataLoader:__tostring__()
    return ("DataLoader: %s (%s task)"):format(self.db_name, self.task)
end


---------------------------------------------------------------------------------------------------

function SetLoader:__init(hdf5_group)
--[[
    Set metadata loader class.

    This class contains several methods to fetch data from a specific
    set (group) in a hdf5 file. It contains useful information about a
    specific group and also several methods to fetch data.

    Parameters
    ----------
    hdf5_group : h5py._hl.group.Group
        hdf5 group object handler.
    set_name : str
        Name of the set.

    Attributes
    ----------
    data : h5py._hl.group.Group
        hdf5 group object handler.
    set : str
        Name of the set.
    fields : tuple
        List of all field names of the set.
    _object_fields : tuple
        List of all field names of the set contained by the 'object_ids' list.
    nelems : int
        Number of rows in 'object_ids'.
]]
    assert(hdf5_group, 'Must input a valid hdf5 group.')
    self.hdf5_group = hdf5_group
    self.set = self:_get_set_name()
    self.fields = self:_get_fields()
    self._object_fields = self:_get_object_fields_data()
    self.nelems = self:_get_num_elements()
    self:_load_hdf5_fields()  -- add all hdf5 datasets as data fields
end

function SetLoader:_get_set_name()
    local str = hdf5._getObjectName(self.hdf5_group._groupID)
    print('\n\n\n\n\n***************')
    print(str)
    print('***************\n\n\n\n\n')
    assert(str, 'No string exists!')
    local str_split = str:split('/')
    return str_split[1]
end

function SetLoader:_get_fields()
    local fields = {}
    for k, v in pairs(self.hdf5_group._children) do
        table.insert(fields, k)
    end
    table.sort(fields)
    return fields
end

function SetLoader:_get_object_fields_data()
    local object_fields_data = self:_get_hdf5_dataset_data('object_fields')
    return string_ascii.convert_ascii_to_str(object_fields_data)
end

function SetLoader:_get_hdf5_dataset_data(name)
    local hdf5_dataset = self:_get_hdf5_dataset(name)
    return hdf5_dataset:all()
end

function SetLoader:_get_hdf5_dataset(name)
    return self.hdf5_group:getOrCreateChild(name)
end

function SetLoader:_get_num_elements()
    local hdf5_dataset = self:_get_hdf5_dataset('object_ids')
    local size = hdf5_dataset:dataspaceSize()
    return size[1]
end

function SetLoader:_load_hdf5_fields()
    for _, field in pairs(self.fields) do
        local obj_id = get_value_id_in_list(field, self._object_fields)
        local hdf5_dataset = self:_get_hdf5_dataset(field)
        self[field] = dbcollection.FieldLoader(hdf5_dataset, obj_id)
    end
end

function SetLoader:get(field, idx)
--[[
    Retrieves data from the dataset's hdf5 metadata file.

    This method retrieves the i'th data from the hdf5 file with the
    same 'field' name. Also, it is possible to retrieve multiple values
    by inserting a list/tuple of number values as indexes.

    Parameters
    ----------
    field : str
        Field name.
    idx : int/list/tuple, optional
        Index number of the field. If it is a list, returns the data
        for all the value indexes of that list.

    Returns
    -------
    np.ndarray
        Tensor array containing the field's data.
    Table
        Table of tensors if using a list of indexes.
]]
    assert(field, ('Must input a valid field name: %s'):format(field))
    assert(is_val_in_table(field, self.fields), ('Field \'%s\' does not exist in the \'%s\' set.')
                                                :format(field, self.set))
    return self[field]:get(idx)
end

function SetLoader:object(idx, convert_to_value)
--[[
    Retrieves a list of all fields' indexes/values of an object composition.

    Retrieves the data's ids or contents of all fields of an object.

    It basically works as calling the get() method for each individual field
    and then groups all values into a list w.r.t. the corresponding order of
    the fields.

    Parameters
    ----------
    idx : int/list/tuple, optional
        Index number of the field. If it is a list, returns the data
        for all the value indexes of that list. If no index is used,
        it returns the entire data field array.
    convert_to_value : bool, optional
        If False, outputs a list of indexes. If True,
        it outputs a list of arrays/values instead of indexes.

    Returns
    -------
    list
        Returns a list of indexes or, if convert_to_value is True,
        a list of data arrays/values.
]]
    local indexes = self:_get_object_indexes(idx)
    if convert_to_value then
        indexes = self._convert(indexes)
    end
    return indexes
end

function SetLoader:_get_object_indexes(idx)
    self:_validate_object_idx(idx)
    return self:get('object_ids', idx)
end

function SetLoader:_validate_object_idx(idx)
    if idx then
        if type(idx) == 'number' then
            assert(idx >= 1, ('idx must be >=1: %d'):format(idx))
        elseif type(idx) == 'table' then
            assert(self:_is_greater_than_zero(idx), ('Table must have indexes >= 1.'))
        else
            error(('Must insert a table or number as input: %s'):format(type(idx)))
        end
    end
end

function SetLoader:_is_greater_than_zero(idx)
    local min = 1
    for k, v in pairs(idx) do
        min = math.min(min, v)
    end
    return min == 1
end

function SetLoader:_convert(idx)
    --[[
        Retrieve data from the dataset's hdf5 metadata file in the original format.

        This method fetches all indices of an object(s), and then it looks up for the
        value for each field in 'object_ids' for a certain index(es), and then it
        groups the fetches data into a single list.

        Parameters
        ----------
        idx : int/table
            Index number of the field. If it is a list, returns the data
            for all the indexes of that list as values.

        Returns
        -------
        str/int/table
            Value/list of a field from the metadata cache file.
    ]]
    local indexes = self:_convert_to_table_of_tables(idx)
    local object_fields = self:_convert_fields_idx_to_val(indexes)
    return self:_convert_fields_output(object_fields)
end

function SetLoader:_convert_to_table_of_tables(idx)
    if type(idx[1]) == 'number' then
        return {idx}
    else
        return idx
    end
end

function SetLoader:_convert_fields_idx_to_val(indexes)
    local output = {}
    for i=1, #indexes do
        local data = self:_get_object_fields_data_from_idx(indexes[i])
        table.insert(output, data)
    end
end

function SetLoader:_get_object_fields_data_from_idx(indexes)
    local data = {}
    for k, field in ipairs(self.object_fields) do
        if indexes[k] > 0 then
            table.insert(data, self:get(field, indexes[k]))
        else
            table.insert(data, {})
        end
    end
    return data
end

function SetLoader:_convert_fields_output(object_fields_values)
    if #object_fields_values > 1 then
        return object_fields
    else
        return object_fields[1]
    end
end

function SetLoader:size(field)
--[[
    Size of a field.

    Returns the number of the elements of a field.

    Parameters
    ----------
    field : str, optional
        Name of the field in the metadata file.

    Returns
    -------
    list
        Returns the size of a field.
]]
    local field = field or 'object_ids'
    assert(is_val_in_table(field, self._object_fields),
           ('Field \'%s\' does not exist in the \'%s\' set.')
           :format(field, self.set))
    return self.hdf5_group:getOrCreateChild(field):dataspaceSize()
end

function SetLoader:list()
--[[
    List of all field names.

    Returns
    -------
    list
        List of all data fields of the dataset.
]]
    return self.fields
end

function SetLoader:object_field_id(field)
    self:_validate_object_field_id_input(field)
    local idx = self:_get_object_field_id(field)
    assert(idx, 'Field does not exist in \'_object_fields\'')
    return idx
end

function SetLoader:_validate_object_field_id_input()
    assert(field, 'Must input a valid field.')
    assert(is_val_in_table(field, self._object_fields),
           ('Field \'%s\' does not exist \'object_fields\' set.')
           :format(field, self.set))
end

function SetLoader:_get_object_field_id(field)
    for i=1, #self._object_fields do
        if field == self._object_fields[i] then
            return i
        end
    end
    return nil
end

function SetLoader:info()
--[[
    Prints information about the data fields of a set.

    Displays information of all fields available like field name,
    size and shape of all sets. If a 'set_name' is provided, it
    displays only the information for that specific set.

    This method provides the necessary information about a data set
    internals to help determine how to use/handle a specific field.
]]
    print(('\n> Set: {}'):format(self.set))
    self:_set_fields_info()
    self:_print_fields_info()
    self:_print_lists_info()
end

function SetLoader:_set_fields_info()
    if self._fields_info == nil then
        self:_init_info_vars()
        self:_set_info_data()
    end
end

function SetLoader:_init_info_vars()
    self._fields_info = {}
    self._lists_info = {}
    self._sizes_info = self:_init_max_sizes()
end

function SetLoader:_init_max_sizes()
    return {
        name = 0,
        shape = 0,
        type = 0,
        name_list = 0,
        shape_list = 0
    }
end

function SetLoader:_set_info_data()
    for i=1, #self.fields do
        self:_set_field_data(self.fields[i])
    end
end

function SetLoader:_set_field_data(field)
    assert(field)
    local name, shape, dtype, name_lists, shape_lists = self:_get_field_metadata(field)
    self:_set_max_sizes_info(#name, #shape, #dtype, #name_lists, #shape_lists)
end

function SetLoader:_get_field_metadata(name)
    assert(name)
    local size, shape, dtype = self:_get_field_size_shape_type(field)
    local name_lists, shape_lists = self:_set_field_metadata(field, shape, dtype)
    return size, shape, dtype, name_lists, shape_lists
end

function SetLoader:_get_field_size_shape_type(field_name)
    assert(field_name)
    local hd5_dataset = self:_get_hdf5_dataset(field_name)
    local size = hd5_dataset:dataspaceSize()
    local shape = get_data_shape(size)
    local dtype = get_data_type_hdf5(hd5_dataset, size)
    return size, shape, dtype
end

function SetLoader:_set_field_metadata(field, shape, dtype)
    assert(field)
    assert(shape)
    assert(dtype)
    local name_lists, shape_lists
    if self:_is_field_a_list(field) then
        name_lists, shape_lists = self:_set_field_metadata_list(field, shape, dtype)
    else
        name_lists, shape_lists = self:_set_field_metadata_default(field, shape, dtype)
    end
    return name_lists, shape_lists
end

function SetLoader:_is_field_a_list(field)
    assert(field)
    if field:match('list_') then
        return true
    else
        return false
    end
end

function SetLoader:_set_field_metadata_list(field, shape, dtype)
    assert(field)
    assert(shape)
    assert(dtype)
    table.insert(self._lists_info, {
        name = field,
        shape = ('shape = %s'):format(shape),
        type = ('dtype = %s'):format(dtype)
    })
    return '', ''
end

function SetLoader:_set_field_metadata_default(field, shape, dtype)
    assert(field)
    assert(shape)
    assert(dtype)
    local s_obj = ''
    if is_val_in_table(field, self._object_fields) then
        s_obj = ("(in 'object_ids', position = {})"):format(self.object_field_id(field))
    end
    table.insert(self._fields_info, {
        name = field,
        shape = ('shape = %s'):format(shape),
        type = ('dtype = %s'):format(dtype),
        obj = s_obj
    })
    return field, dtype
end

function SetLoader:_set_max_sizes_info(size_name, size_shape, size_dtype, size_name_lists, size_shape_lists)
    assert(size_name)
    assert(size_shape)
    assert(size_dtype)
    assert(size_name_lists)
    assert(size_shape_lists)
    self._sizes_info.name = math.max(self._sizes_info.name, size_name)
    self._sizes_info.shape = math.max(self._sizes_info.hape, size_shape)
    self._sizes_info.type = math.max(self._sizes_info.type, size_dtype)
    self._sizes_info.name_list = math.max(self._sizes_info.name_list, size_name_lists)
    self._sizes_info.shape_list = math.max(self._sizes_info.shape_list, size_shape_lists)
end

function SetLoader:_print_fields_info()
    for i=1, #self._fields_info do
        local field_info = self._fields_info[i]
        local s_name = self:_get_field_name_str(i)
        local s_shape = self:_get_field_shape_str(i)
        local s_obj = self._fields_info[i]["obj"]
        local s_type = self:_get_field_type_str(i, #s_obj > 0)
        print(s_name .. s_shape .. s_type .. s_obj)
    end
end

function SetLoader:_get_field_name_str(idx)
    assert(idx)
    local offset = 8
    local name = self._fields_info[idx]['name']
    local max_size = self._sizes_info.name
    local str_padding = string.rep(' ', max_size + offset - #name)
    return ('   - %s, '):format(name .. str_padding)
end

function SetLoader:_get_field_shape_str(idx)
    assert(idx)
    local offset = 3
    local shape = self._fields_info[idx]["shape"]
    local max_size = self._sizes_info.shape
    local str_padding = string.rep(' ', max_size + offset - #shape)
    return ('%s, '):format(shape .. str_padding)
end

function SetLoader:_get_field_type_str(idx, is_comma)
    assert(idx)
    assert(is_comma ~= nil)
    local offset = 3
    local dtype = self._fields_info[idx]["type"]
    local comma = ''
    if is_comma then
        comma = ','
    end
    local str_padding = string.rep(' ', self._sizes_info.type + offset - #dtype)
    return fields_info["type"] .. comma .. str_padding
end

function SetLoader:_print_lists_info()
    if #self._lists_info > 0 then
        print('\n   (Pre-ordered lists)')
        for i=1, #lists_info do
            local s_name = self:_get_list_name_str(idx)
            local s_shape = self:_get_list_shape_str(idx)
            local s_type = self._lists_info[idx]["type"]
            print(s_name .. s_shape .. s_type)
        end
    end
end

function SetLoader:_get_list_name_str(idx)
    assert(idx)
    local offset = 8
    local name = self._lists_info[idx]["name"]
    local str_padding = string.rep(' ', self._sizes_info.name_list + offset - #name)
    return ('   - %s, '):format(name .. str_padding)
end

function SetLoader:_get_list_shape_str(idx)
    assert(idx)
    local offset = 3
    local shape = self._lists_info[idx]["shape"]
    local str_padding = string.rep(' ', self._sizes_info.shape_list + offset - #shape)
    return ('%s, '):format(shape .. str_padding)
end

function SetLoader:__len__()
    return self.nelems
end

function SetLoader:__tostring__()
    return ('SetLoader: set<%s>, len<%d>'):format(self.set, self.nelems)
end


---------------------------------------------------------------------------------------------------

function FieldLoader:__init(hdf5_field, obj_id)
--[[
    Field metadata loader class.

    This class contains several methods to fetch data from a specific
    field of a set (group) in a hdf5 file. It contains useful information
    about the field and also several methods to fetch data.

    Parameters
    ----------
    hdf5_field : h5py._hl.dataset.Dataset
        hdf5 field object handler.

    Attributes
    ----------
    data : h5py._hl.dataset.Dataset
        hdf5 group object handler.
    set : str
        Name of the set.
    name : str
        Name of the field.
    type : type
        Type of the field's data.
    shape : tuple
        Shape of the field's data.
    fillvalue : int
        Value used to pad arrays when storing the data in the hdf5 file.
    obj_id : int
        Identifier of the field if contained in the 'object_ids' list.
]]
    assert(hdf5_field, 'Must input a valid hdf5 dataset.')
    self.data = hdf5_field
    self.hdf5_handler = hdf5_field
    self._in_memory = false
    self.set = self:_get_set_name()
    self.name = self:_get_field_name()
    self.size = self:_get_field_size()
    self.shape = get_data_shape(self.size)
    self.type = get_data_type_hdf5(self.data, self.size)
    self.ids_list = self:_get_ids_list()
    self.ndims = #self.size
    -- fillvalue not implemented in hdf5 lib
    self.obj_id = obj_id
end

function FieldLoader:_get_set_name()
    local hdf5_object_str = self:_get_hdf5_object_str()
    return hdf5_object_str:split('/')[1]
end

function FieldLoader:_get_field_name()
    local hdf5_object_str = self:_get_hdf5_object_str()
    return hdf5_object_str:split('/')[2]
end

function FieldLoader:_get_hdf5_object_str()
    return hdf5._getObjectName(self.hdf5_handler._datasetID)
end

function FieldLoader:_get_field_size()
    return self.data:dataspaceSize()
end

function FieldLoader:_get_ids_list()
    local ids = {}
    for i=1, #self.size do
        table.insert(ids, {1, self.size[i]})
    end
    return ids
end

function FieldLoader:get(idx)
--[[
    Retrieves data of the field from the dataset's hdf5 metadata file.

    This method retrieves the i'th data from the hdf5 file. Also, it is
    possible to retrieve multiple values by inserting a list/tuple of
    number values as indexes.

    Parameters
    ----------
    idx : number/table, optional
        Index number of he field. If it is a list, returns the data
        for all the value indexes of that list.

    Returns
    -------
    torch.*Tensor
        Numpy array containing the field's data.
    table
        List of tensors if using a list of indexes.
]]
    assert(idx, 'Must input a number or table as input.')
    assert(type(idx) == 'number' or dtype == 'table', ('Must input a number or table as input: %s.'):format(dtype))
    local data = {}
    if idx then
        return self:_get_range(idx)
    else
        return self:_get_all()
    end
    return data
end

function FieldLoader:_get_range(idx)
    assert(idx)
    if type(idx) == 'number' then
        return self:_get_data_single_id(idx)
    else
        return self:_get_data_multiple_id(idx)
    end
end

function FieldLoader:_get_data_single_id(idx)
    if self._in_memory then
        return self:_get_data_memory(idx)
    else
        return self:_get_data_hdf5(idx)
    end
end

function FieldLoader:_get_data_memory(idx)
    assert(idx)
    return self.data[idx]
end

function FieldLoader:_get_data_hdf5(idx)
    assert(idx)
    local id = self:_get_id(idx)
    return self.data:partial(unpack(id))
end

function FieldLoader:_get_ids(idx)
    assert(idx)
    if type(idx) == 'number' then
        return self:_get_ids_single(idx)
    else
        return self:_get_ids_multiple(idx)
    end
end

function FieldLoader:_get_ids_single(idx)
    assert(idx)
    local ids = self.ids_list
    ids[1][1] = idx
    ids[1][2] = idx
    return ids
end

function FieldLoader:_get_ids_multiple(idx)
    assert(idx)
    assert(#idx > self.ndims, ('too many indices provided: got %d, max expected %d')
                              :format(#idx, self.ndims))
    local ids = self.ids_list
    for i=1, #idx do
        if type(idx[i]) == 'number' then
            ids[i][1] = idx[i]
            ids[i][2] = idx[i]
        else
            ids[i][1] = idx[i][1] or ids[i][1]
            ids[i][2] = idx[i][2] or ids[i][2]
        end
    end
    return ids
end

function FieldLoader:_get_data_multiple_id(idx)
--[[
    Returns a single tensor where the first dim rows corresponds to the indexes.
]]
    assert(idx)
    local data
    for i=1, #idx do
        local sample = self:_get_data_single_id(idx[i])
        if i > 1 then
            data = data:cat(sample, 1)
        else
            data = sample
        end
    end
    return data
end

function FieldLoader:_get_all()
    if self._in_memory then
        return self.data
    else
        return self.data:all()
    end
end

function FieldLoader:size()
--[[
    Size of the field.

    Returns the number of the elements of the field.

    Returns
    -------
    table
        Returns the size of the field.
]]
    return self.size
end

function FieldLoader:object_field_id()
--[[
    Retrieves the index position of the field in the 'object_ids' list.

    This method returns the position of the field in the 'object_ids' object.
    If the field is not contained in this object, it returns a null value.

    Returns
    -------
    int
        Index of the field in the 'object_ids' list.
]]
    return self.obj_id
end

function FieldLoader:info()
--[[
    Prints information about the field.

    Displays information like name, size and shape of the field.
]]
    if self.obj_id then
        print(('Field: %s,  shape = %s,  dtype = %s,  (in \'object_ids\', position = %d)')
              :format(self.name, str(self.shape), str(self.type), self.obj_id))
    else
        print(('Field: %s,  shape = %s,  dtype = %s')
              :format(self.name, str(self.shape), str(self.type)))
    end
end

function FieldLoader:_set_to_memory(is_in_memory)
--[[
    Stores the contents of the field in a numpy array if True.

    Parameters
    ----------
    is_in_memory : bool
        Move the data to memory (if True).
]]
    if is_in_memory then
        self.data = self.hdf5_handler:all()
    else
        self.data = self.hdf5_handler
    end
    self._in_memory = is_in_memory
end

function FieldLoader:to_memory(is_to_memory)
--[[
    Modifies how data is accessed and stored.

    Accessing data from a field can be done in two ways: memory or disk.
    To enable data allocation and access from memory requires the user to
    specify a boolean. If set to True, data is allocated to a numpy ndarray
    and all accesses are done in memory. Otherwise, data is kept in disk and
    accesses are done using the HDF5 object handler.

    Parameters
    ----------
    is_to_memory : bool
        Move the data to memory (if True).
]]
    self:_set_to_memory(is_to_memory)
end

function FieldLoader:__len__()
    return self.size
end

function FieldLoader:__tostring__()
    local str
    if self._in_memory then
        str = ('FieldLoader: <torch.*Tensor "%s": shape %s, type "%s">')
              :format(self.name, self.shape, self.type)
    else
        str = ('FieldLoader: <HDF5File "%s": shape %s, type "%s">')
              :format(self.name, self.shape, self.type)
    end
    return str
end

function FieldLoader:__index__(idx)
    assert(idx, 'Error: must input a non-empty index')
    -- vvv **temporary fix** vvv
    -- see https://github.com/torch/torch7/blob/a2873a95a500e03c8f7eeb363cdb7058cc297f5b/lib/luaT/README.md#operator-overloading
    if type(idx) == 'string' then
        return false
    end
    -- ^^^ **temporary fix** ^^^
    if self._in_memory then
        return self:_get_data_memory(idx), true
    else
        return self:_get_data_hdf5(idx), true
    end
end
