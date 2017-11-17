--[[
    Dataset's metadata loader classes.
--]]

local hdf5 = require 'hdf5'
local dbcollection = require 'dbcollection.env'
local string_ascii = require 'dbcollection.utils.string_ascii'

local DataLoader = torch.class('dbcollection.DatasetLoader', dbcollection)
local SetLoader = torch.class('dbcollection.SetLoader', dbcollection)
local FieldLoader = torch.class('dbcollection.FieldLoader', dbcollection)

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
    self.file = hdf5.open(self.hdf5_filepath, 'r')
    self.root_path = '/'

    -- make links for all groups (train/val/test/etc) for easier access
    self.sets = {}
    self.object_fields = {}

    -- make links for all groups (train/val/test/etc) for easier access
    self.sets = {}
    self.object_fields = {}
    local group_default = self.file:read(self.root_path)
    for k, v in pairs(group_default._children) do
        -- add set to the table
        table.insert(self.sets, k)

        self['k'] = dbcollection.SetLoader(self.file:read(self.root_path + k), k)

        -- fetch list of field names that compose the object list.
        local data = self.file:read(('%s/%s/object_fields'):format(self.root_path, k)):all()
        if data:dim()==1 then data = data:view(1,-1) end
        self.object_fields[k] = string_ascii.convert_ascii_to_str(data)
    end
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
        assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                    :format(set_name))
        return self[set_name]:size(field)
    else
        local out = {}
        for set_name in pairs(self.sets) do
            out[set_name] = self[set_name]:size(field)
        end
        return out
    end
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
        assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                    :format(set_name))
        return self[set_name]:list()
    else
        local out = {}
        for set_name in pairs(self.sets) do
            out[set_name] = self[set_name]:list()
        end
        return out
    end
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
        assert(self.sets[set_name], ('Set %s does not exist for this dataset.')
                                    :format(set_name))
        self[set_name]:info()
    else
        for set_name in pairs(self.sets) do
            self[set_name]:info()
        end
    end
end

function DataLoader:__tostring()
    return ("Dataloader: %s (%s task)"):format(self.db_name, self.task)
end

---------------------------------------------------------------------------------------------------

function SetLoader:__init(hdf5_group, set_name)
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
    assert(set_name, 'Must input a valid hdf5 group.')
    self.data = hdf5_group
    self.set = set_name
    self.fields = tuple(hdf5_group.keys())

    self.fields = {}
    for k, v in pairs(self.data._children) do
        table.insert(self.fields, k)
    end
    table.sort(self.fields)
    local object_fields_data = self.data:getOrCreateChild('object_fields'):all()
    self._object_fields = string_ascii.convert_ascii_to_str(object_fields_data)
    self.nelems = self.data:getOrCreateChild('object_fields'):dataspaceSize()[1]

    local function fetch_id_in_list(val, list)
        for i=1, #list do
            if list[i] == val then
                return i
            end
        end
        return {}
    end

    -- add fields to the class
    for field in pairs(self.fields) do
        local obj_id = fetch_id_in_list(field, self._object_fields)
        self[field] = FieldLoader(self.data:getOrCreateChild(field), obj_id)
    end
end

local function val_in_table(val, t)
    for k, v in pairs(t) do
        if v == vall then
            return true
        end
    end
    return false
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
    assert(val_in_table(field, self.fields), ('Field \'%s\' does not exist in the \'%s\' set.')
                                             :format(field, self.set))
    return self[field]:get(idx)
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
    local indexes = idx
    if type(idx[1]) == 'number' then
        indexes = {idx}
    end

    local output = {}
    for i=1, #idx do
        local data = {}
        for k, field in ipairs(self.object_fields) do
            if indexes[i][k] > 0 then
                table.insert(data, self:get(field, indexes[i][k]))
            else
                table.insert(data, {})
            end
        end
        table.insert(out, data)
    end
    if #output > 1 then
        return output
    else
        return output[1]
    end
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
    if idx then
        if type(idx) == 'number' then
            assert(idx >= 1, ('idx must be >=1: %d'):format(idx))
        elseif type(idx) == 'table' then
            local min = 1
            for k, v in pairs(idx) do
                min = math.min(min, v)
            end
            assert(min==1, ('Table must have indexes >= 1.'))
        else
            error(('Must insert a table or number as input: %s':format(type(idx)))
        end
    end

    local indexes = self:get('object_ids', idx)

    if convert_to_value then
        return self._convert(indexes)
    else
        return indexes
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
    assert(val_in_table(field, self._object_fields),
           ('Field \'%s\' does not exist in the \'%s\' set.')
           :format(field, self.set))
    return self.data:getOrCreateChild(field):dataspaceSize()
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
    assert(field, 'Must input a valid field.')
    assert(val_in_table(field, self._object_fields),
           ('Field \'%s\' does not exist \'object_fields\' set.')
           :format(field, self.set))
    local idx
    for i=1, #self._object_fields do
        if field == self._object_fields[i] then
            return i
        end
    end
    error('Field does not exist in \'_object_fields\'')
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

    local fields_info = {}
    local lists_info = {}
    local maxsize_name = 0
    local maxsize_shape = 0
    local maxsize_name_lists = 0
    local maxsize_shape_lists = 0
    local maxsize_type = 0
    for i=1, #self.fields do
        local f = self:get(field)
        local size = f:size():totable()
        local shape="("
        for j=1, #size do
            shape = shape .. size[j]
            if j < #size then
                shape = shape .. ', '
            else
                shape = shape .. ')'
            end
        end
        local dtype = torch.type(f)
        if fields:match('list_') then
            table.insert(lists_info, {
                name = field,
                shape = ('shape = %s'):format(shape),
                type = ('dtype = %s'):format(dtype)
            })
        else
            local s_obj = ''
            if val_in_table(field, self._object_fields) then
                s_obj = ("(in 'object_ids', position = {})")
                        :format(self.object_field_id(field))
            end
            table.insert(fields_info, {
                name = field,
                shape = ('shape = %s'):format(shape),
                type = ('dtype = %s'):format(dtype),
                obj = s_obj
            })
            maxsize_name_lists = math.max(maxsize_name_lists, #field)
            maxsize_shape_lists = math.max(maxsize_shape_lists, #dtype)
        end

        maxsize_name = math.max(maxsize_name, #name)
        maxsize_shape = math.max(maxsize_shape, #shape)
        maxsize_type = math.max(maxsize_type, #dtype)
    end

    for i=1, #fields_info do
        local s_name = ('   - %s, '):format(fields_info["name"] .. string.rep(' ', maxsize_name + 8 - #fields_info["name"]))
        local s_shape = ('%s, '):format(fields_info["shape"] .. string.rep(' ', maxsize_shape + 3 - #fields_info["shape"]))
        local s_obj = fields_info["obj"]
        local comma = ''
        if #s_obj > 0 then
            comma = ','
        end
        local s_type = fields_info["type"] .. comma .. string.rep(' ', maxsize_type + 3 - #fields_info["type"])
        print(s_name .. s_shape .. s_type .. s_obj)
    end

    if #lists_info > 0 then
        print('\n   (Pre-ordered lists)')

        for i=1, #lists_info do
            local s_name = ('   - %s, '):format(lists_info["name"] .. string.rep(' ', maxsize_name_list + 8 - #lists_info["name"]))
            local s_shape = ('%s, '):format(lists_info["shape"] .. string.rep(' ', maxsize_shape_list + 3 - #lists_info["shape"]))
            local s_type = lists_info["type"]
            print(s_name .. s_shape .. s_type)
        end
    end
end

function SetLoader:__tostring()
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

end

function FieldLoader:get(idx)
end

function FieldLoader:size()
end

function FieldLoader:object_field_id()
end

function FieldLoader:info()
end

function FieldLoader:_set_to_memory()
end

function FieldLoader:_get_to_memory()
end

function FieldLoader:__tostring()
end

function FieldLoader:__tostring__()
end

function FieldLoader:__len()
end

function FieldLoader:__len__()
end

function FieldLoader:__index()
end

function FieldLoader:__index__()
end
