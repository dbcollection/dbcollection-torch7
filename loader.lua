--[[
    Dataset's metadata loader classes.
--]]

local argcheck = require 'argcheck'

local hdf5 = require 'hdf5'
local dbcollection = require 'dbcollection.env'
local string_ascii = require 'dbcollection.utils.string_ascii'

local DataLoader = torch.class('dbcollection.DataLoader', dbcollection)
local SetLoader = torch.class('dbcollection.SetLoader', dbcollection)
local FieldLoader = torch.class('dbcollection.FieldLoader', dbcollection)


---------------------------------------------------------------------------------------------------

--[[ Split a string w.r.t. a single or a sequence of characters ]]
local function split_str(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={}
    local i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

local function get_value_id_in_list(val, list)
    for i=1, #list do
        if list[i] == val then
            return i
        end
    end
    return nil
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

function DataLoader:__init(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
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

        ]],
        {name="name", type="string",
         help="Name of the dataset."},
        {name="task", type="string",
         help="Name of the task."},
        {name="data_dir", type="string",
         help="Path of the dataset's data directory on disk."},
        {name="hdf5_filepath", type="string",
         help="Path of the metadata cache file stored on disk."}
    }

    local args = initcheck(...)

    -- store information of the dataset
    self.db_name = args.name
    self.task = args.task
    self.data_dir = args.data_dir
    self.hdf5_filepath = args.hdf5_filepath

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
        self[set] = dbcollection.SetLoader(self:_get_hdf5_group(hdf5_group_path))
    end
end

function DataLoader:_get_hdf5_group(path)
    return self.file:read(path)
end

function DataLoader:get(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves data from the dataset's hdf5 metadata file.

            This method retrieves the i'th data from the hdf5 file with the
            same 'field' name. Also, it is possible to retrieve multiple values
            by inserting a list/tuple of number values as indexes.

            Parameters
            ----------
            set : str
                Name of the set.
            field : str
                Name of the field.
            index : number/table, optional
                Index number of the field. If it is a list, returns the data
                for all the value indexes of that list.

            Returns
            -------
            torch.*Tensor
                Numpy array containing the field's data.
        ]],
        {name="set", type="string",
         help="Name of the set."},
        {name="field", type="string",
         help="Name of the field."},
        {name="index", type="table", default={},
         help="Index number of the field. If it is a list, returns the data " ..
              "for all the value indexes of that list.",
         opt = true}
    }

    -- Workaround to manage have multiple types for the same input.
    -- First the input checks if it is a number. If the input arg
    -- is not a number, do a second argument parsing to check if the
    -- second type matches the input argument.
    local initcheck_ = argcheck{
        quiet=true,
        pack=true,
        {name="set", type="string"},
        {name="field", type="string"},
        {name="index", type="number"}
    }

    local status, args = initcheck_(...)

    if not status then
        args = initcheck(...)
    end

    self:_check_if_set_is_valid(args.set)
    return self[args.set]:get(args.field, args.index)
end

function DataLoader:_check_if_set_is_valid(set)
    local is_set_name_valid = is_val_in_table(set, self.sets)
    assert(is_set_name_valid, ('Set %s does not exist for this dataset.'):format(set))
end

function DataLoader:object(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves a list of all fields' indexes/values of an object composition.

            Retrieves the data's ids or contents of all fields of an object.

            It basically works as calling the get() method for each individual field
            and then groups all values into a list w.r.t. the corresponding order of
            the fields.

            Parameters
            ----------
            set : str
                Name of the set.
            index : number/table, optional
                Index number of the field. If it is a list, returns the data
                for all the value indexes of that list. If no index is used,
                it returns the entire data field array.
            convert_to_value : bool, optional
                If False, outputs a list of indexes. If True,
                it outputs a list of arrays/values instead of indexes.

            Returns
            -------
            table
                Returns a list of indexes or, if convert_to_value is True,
                a list of data arrays/values.
        ]],
        {name="set", type="string",
         help="Name of the set."},
        {name="index", type="table", default={},
         help="Index number of the field. If it is a list, returns the data " ..
              "for all the value indexes of that list.",
         opt=true},
        {name="convert_to_value", type="boolean", default=false,
         help="If False, outputs a list of indexes. If True, " ..
              "it outputs a list of arrays/values instead of indexes.",
         opt=true}
    }

    -- Workaround to manage have multiple types for the same input.
    -- First the input checks if it is a number. If the input arg
    -- is not a number, do a second argument parsing to check if the
    -- second type matches the input argument.
    local initcheck_ = argcheck{
        quiet=true,
        pack=true,
        {name="set", type="string"},
        {name="index", type="number"},
        {name="convert_to_value", type="boolean", default=false, opt=true}
    }

    local status, args = initcheck_(...)

    if not status then
        args = initcheck(...)
    end

    self:_check_if_set_is_valid(args.set)
    return self[args.set]:object(args.index, args.convert_to_value)
end

function DataLoader:size(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Size of a field.

            Returns the number of the elements of a field.

            Parameters
            ----------
            set : str, optional
                Name of the set.
            field : str, optional
                Name of the field in the metadata file.

            Returns
            -------
            table
                Returns the size of a field.
        ]],
        {name="set", type="string",
         help="Name of the set.",
         opt=true},
        {name="field", type="string", default='object_ids',
         help="Name of the field in the metadata file.",
         opt = true}
    }

    local args = initcheck(...)

    if args.set then
        return self:_get_set_size(args.set, args.field)
    else
        return self:_get_set_size_all(args.field)
    end
end

function DataLoader:_get_set_size(set, field)
    assert(field, 'Must input a field')
    self:_check_if_set_is_valid(set)
    return self[set]:size(field)
end

function DataLoader:_get_set_size_all(field)
    assert(field, 'Must input a field')
    local out = {}
    for _, set_name in pairs(self.sets) do
        out[set_name] = self:_get_set_size(set_name, field)
    end
    return out
end

function DataLoader:list(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            List of all field names of a set.

            Parameters
            ----------
            set : str, optional
                Name of the set.

            Returns
            -------
            table
                List of all data fields of the dataset.
        ]],
        {name="set", type="string",
         help="Name of the set.",
         opt=true}
    }

    local args = initcheck(...)

    if args.set then
        return self:_get_set_list(args.set)
    else
        return self:_get_set_list_all()
    end
end

function DataLoader:_get_set_list(set)
    self:_check_if_set_is_valid(set)
    return self[set]:list()
end

function DataLoader:_get_set_list_all()
    local out = {}
    for _, set_name in pairs(self.sets) do
        out[set_name] = self[set_name]:list()
    end
    return out
end

function DataLoader:object_field_id(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves the index position of a field in the 'object_ids' list.

            This method returns the position of a field in the 'object_ids' object.
            If the field is not contained in this object, it returns a null value.

            Parameters
            ----------
            set : str
                Name of the set.
            field : str
                Name of the field in the metadata file.

            Returns
            -------
            number
                Index of the field in the 'object_ids' list.
        ]],
        {name="set", type="string",
         help="Name of the set."},
        {name="field", type="string",
         help="Name of the field in the metadata file."}
    }

    local args = initcheck(...)

    self:_check_if_set_is_valid(args.set)
    return self[args.set]:object_field_id(args.field)
end

function DataLoader:info(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
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
            set : str, optional
                Name of the set.
        ]],
        {name="set", type="string",
         help="Name of the set.",
         opt=true}
    }

    local args = initcheck(...)

    if args.set then
        self:_get_set_info(args.set)
    else
        self:_get_set_info_all()
    end
end

function DataLoader:_get_set_info(set)
    self:_check_if_set_is_valid(set)
    self[set]:info()
end

function DataLoader:_get_set_info_all()
    for _, set_name in pairs(self.sets) do
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

function SetLoader:__init(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Set metadata loader class.

            This class contains several methods to fetch data from a specific
            set (group) in a hdf5 file. It contains useful information about a
            specific group and also several methods to fetch data.

            Parameters
            ----------
            hdf5_group : hdf5.HDF5Group
                hdf5 group object handler.

            Attributes
            ----------
            data : hdf5.HDF5Group
                hdf5 group object handler.
            set : str
                Name of the set.
            fields : tuple
                List of all field names of the set.
            _object_fields : tuple
                List of all field names of the set contained by the 'object_ids' list.
            nelems : int
                Number of rows in 'object_ids'.
        ]],
        {name="hdf5_group", type="hdf5.HDF5Group",
         help="hdf5 group object handler."}
    }

    local args = initcheck(...)

    self.hdf5_group = args.hdf5_group
    self.set = self:_get_set_name()
    self.fields = self:_get_fields()
    self._object_fields = self:_get_object_fields_data()
    self.nelems = self:_get_num_elements()
    self:_load_hdf5_fields()  -- add all hdf5 datasets as data fields
end

function SetLoader:_get_set_name()
    local hdf5_object_str = hdf5._getObjectName(self.hdf5_group._groupID)
    local str = split_str(hdf5_object_str, '/')
    return str[1]
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
    local output = string_ascii.convert_ascii_to_str(object_fields_data)
    if type(output) == 'string' then
        output = {output}
    end
    return output
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

function SetLoader:get(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves data from the dataset's hdf5 metadata file.

            This method retrieves the i'th data from the hdf5 file with the
            same 'field' name. Also, it is possible to retrieve multiple values
            by inserting a list/tuple of number values as indexes.

            Parameters
            ----------
            field : str
                Field name.
            index : number/table, optional
                Index number of the field. If it is a list, returns the data
                for all the value indexes of that list.

            Returns
            -------
            torch.*Tensor
                Tensor array containing the field's data.
        ]],
        {name="field", type="string",
         help="Name of the dataset."},
        {name="index", type="table", default={},
         help="Index number of the field. If it is a list, returns the data " ..
             "for all the value indexes of that list.",
         opt=true}
    }

    -- Workaround to manage have multiple types for the same input.
    -- First the input checks if it is a number. If the input arg
    -- is not a number, do a second argument parsing to check if the
    -- second type matches the input argument.
    local initcheck_ = argcheck{
        quiet=true,
        pack=true,
        {name="field", type="string"},
        {name="index", type="number"}
    }

    local status, args = initcheck_(...)

    if not status then
        args = initcheck(...)
    end

    local is_field_valid = is_val_in_table(args.field, self.fields)
    assert(is_field_valid, ('Field \'%s\' does not exist in the \'%s\' set.'):format(args.field, self.set))
    return self[args.field]:get(args.index)
end

function SetLoader:object(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves a list of all fields' indexes/values of an object composition.

            Retrieves the data's ids or contents of all fields of an object.

            It basically works as calling the get() method for each individual field
            and then groups all values into a list w.r.t. the corresponding order of
            the fields.

            Parameters
            ----------
            index : number/table, optional
                Index number of the field. If it is a list, returns the data
                for all the value indexes of that list. If no index is used,
                it returns the entire data field array.
            convert_to_value : bool, optional
                If False, outputs a list of indexes. If True,
                it outputs a list of arrays/values instead of indexes.

            Returns
            -------
            table
                Returns a list of indexes or, if convert_to_value is True,
                a list of data arrays/values.
        ]],
        {name="index", type="table",
         help="Index number of the field.",
         opt=true},
        {name="convert_to_value", type="boolean", default=false,
         help="If False, outputs a list of indexes. If True, " ..
              "it outputs a list of arrays/values instead of indexes.",
         opt=true}
    }

    -- Workaround to manage have multiple types for the same input.
    -- First the input checks if it is a number. If the input arg
    -- is not a number, do a second argument parsing to check if the
    -- second type matches the input argument.
    local initcheck_ = argcheck{
        quiet=true,
        pack=true,
        {name="index", type="number"},
        {name="convert_to_value", type="boolean", opt=true}
    }

    local status, args = initcheck_(...)

    if not status then
        args = initcheck(...)
    end

    local indexes = self:_get_object_indexes(args.index)
    if args.convert_to_value then
        indexes = self:_convert(indexes)
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
    assert(idx)

    local idx_ = idx
    if idx:nDimension() == 1 then
        idx_ = idx_:view(1, -1)
    end

    local output = {}
    local num_samples = idx_:size(1)
    for i=1, num_samples do
        local data = self:_get_object_field_data_from_idx(idx_[i])
        table.insert(output, data)
    end
    return output
end

function SetLoader:_get_object_field_data_from_idx(idx)
    local data = {}
    for k, field in ipairs(self._object_fields) do
        if idx[k] >= 0 then
            -- because python is 0-indexed, we need to increment
            -- the hdf5 data elements by one to get the correct index
            table.insert(data, self:get(field, idx[k] + 1))
        else
            table.insert(data, {})
        end
    end
    return data
end

function SetLoader:size(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Size of a field.

            Returns the number of the elements of a field.

            Parameters
            ----------
            field : str, optional
                Name of the field in the metadata file.

            Returns
            -------
            table
                Returns the size of the field.
        ]],
        {name="field", type="string", default='object_ids',
         help="Name of the dataset.",
         opt=true}
    }

    local args = initcheck(...)

    if args.field ~= 'object_ids' then
        local is_field_valid = is_val_in_table(args.field, self._object_fields)
        assert(is_field_valid, ('Field \'%s\' does not exist in the \'%s\' set.'):format(field, self.set))
    end

    return self[args.field]:size()
end

function SetLoader:list(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            List of all field names.

            Returns
            -------
            list
                List of all data fields of the dataset.
        ]]
    }

    local args = initcheck(...)

    return self.fields
end

function SetLoader:object_field_id(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves the index position of the field in the 'object_ids' list.

            This method returns the position of the field in the 'object_ids' object.
            If the field is not contained in this object, it returns a null value.

            Parameters
            ----------
            field : str
                Name of the field in the metadata file.

            Returns
            -------
            number
                Index of the field in the 'object_ids' list.
        ]],
        {name="field", type="string",
         help="Name of the field in the metadata file."}
    }

    local args = initcheck(...)

    self:_validate_object_field_id_input(args.field)
    local idx = self[args.field]:object_field_id()
    assert(idx, ('Field \'%s\' does not exist in \'_object_fields\''):format(args.field))
    return idx
end

function SetLoader:_validate_object_field_id_input(field)
    assert(field, 'Must input a valid field.')
    assert(is_val_in_table(field, self._object_fields),
           ('Field \'%s\' does not exist \'object_fields\' set.')
           :format(field, self.set))
end

function SetLoader:info(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Prints information about the data fields of a set.

            Displays information of all fields available like field name,
            size and shape of all sets. If a 'set_name' is provided, it
            displays only the information for that specific set.

            This method provides the necessary information about a data set
            internals to help determine how to use/handle a specific field.
        ]]
    }

    local args = initcheck(...)

    print(('\n> Set: %s'):format(self.set))
    self:_set_fields_info()
    self:_print_fields_info()
    self:_print_lists_info()
end

function SetLoader:_set_fields_info()
    if self._fields_info == nil then
        self:_init_info_vars()
        self:_set_info_data()
        self:_set_max_sizes()
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
        shape_list = 0,
        type_list = 0,
    }
end

function SetLoader:_set_info_data()
    for i=1, #self.fields do
        self:_set_field_data(self.fields[i])
    end
end

function SetLoader:_set_field_data(field)
    assert(field)
    if self:_is_field_a_list(field) then
        self:_set_list_info_metadata(field)
    else
        self:_set_field_info_metadata(field)
    end
end

function SetLoader:_is_field_a_list(field)
    assert(field)
    if field:match('list_') then
        return true
    else
        return false
    end
end

function SetLoader:_set_list_info_metadata(field)
    local shape, dtype = self:_get_field_shape_type(field)
    self:_set_list_metadata(field, shape, dtype)
end

function SetLoader:_get_field_shape_type(field)
    assert(field)
    local hd5_dataset = self:_get_hdf5_dataset(field)
    local size = hd5_dataset:dataspaceSize()
    local shape = get_data_shape(size)
    local dtype = get_data_type_hdf5(hd5_dataset, size)
    return shape, dtype
end

function SetLoader:_set_list_metadata(field, shape, dtype)
    assert(field)
    assert(shape)
    assert(dtype)
    table.insert(self._lists_info, {
        name = field,
        shape = ('shape = %s'):format(shape),
        type = ('dtype = %s'):format(dtype)
    })
end

function SetLoader:_set_field_info_metadata(field)
    local shape, dtype = self:_get_field_shape_type(field)
    self:_set_field_metadata(field, shape, dtype)
end

function SetLoader:_set_field_metadata(field, shape, dtype)
    assert(field)
    assert(shape)
    assert(dtype)
    local s_obj = ''
    if is_val_in_table(field, self._object_fields) then
        s_obj = ("(in 'object_ids', position = %d)"):format(self:object_field_id(field))
    end
    table.insert(self._fields_info, {
        name = field,
        shape = ('shape = %s'):format(shape),
        type = ('dtype = %s'):format(dtype),
        obj = s_obj
    })
end

function SetLoader:_set_max_sizes()
    self:_set_max_sizes_fields()
    self:_set_max_sizes_lists()
end

function SetLoader:_set_max_sizes_fields()
    for i=1, #self._fields_info do
        self._sizes_info.name = math.max(self._sizes_info.name, #self._fields_info[i]['name'])
        self._sizes_info.shape = math.max(self._sizes_info.shape, #self._fields_info[i]['shape'])
        self._sizes_info.type = math.max(self._sizes_info.type, #self._fields_info[i]['type'])
    end
end

function SetLoader:_set_max_sizes_lists()
    for i=1, #self._lists_info do
        self._sizes_info.name_list = math.max(self._sizes_info.name_list, #self._lists_info[i]['name'])
        self._sizes_info.shape_list = math.max(self._sizes_info.shape_list, #self._lists_info[i]['shape'])
        self._sizes_info.type_list = math.max(self._sizes_info.type_list, #self._lists_info[i]['type'])
    end
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
    local offset = 1
    local name = self._fields_info[idx]['name']
    local max_size = self._sizes_info.name
    local str_padding = string.rep(' ', max_size + offset - #name)
    return ('   - %s, %s'):format(name, str_padding)
end

function SetLoader:_get_field_shape_str(idx)
    assert(idx)
    local offset = 1
    local shape = self._fields_info[idx]["shape"]
    local max_size = self._sizes_info.shape
    local str_padding = string.rep(' ', max_size + offset - #shape)
    return ('%s, %s'):format(shape, str_padding)
end

function SetLoader:_get_field_type_str(idx, is_comma)
    assert(idx)
    assert(is_comma ~= nil)
    local offset = 1
    local dtype = self._fields_info[idx]["type"]
    local comma = ''
    if is_comma then
        comma = ','
    end
    local str_padding = string.rep(' ', self._sizes_info.type + offset - #dtype)
    return dtype .. comma .. str_padding
end

function SetLoader:_print_lists_info()
    if #self._lists_info > 0 then
        print('\n   (Pre-ordered lists)')
        for i=1, #self._lists_info do
            local s_name = self:_get_list_name_str(i)
            local s_shape = self:_get_list_shape_str(i)
            local s_type = self._lists_info[i]["type"]
            print(s_name .. s_shape .. s_type)
        end
    end
end

function SetLoader:_get_list_name_str(idx)
    assert(idx)
    local offset = 1
    local name = self._lists_info[idx]["name"]
    local str_padding = string.rep(' ', self._sizes_info.name_list + offset - #name)
    return ('   - %s, %s'):format(name, str_padding)
end

function SetLoader:_get_list_shape_str(idx)
    assert(idx)
    local offset = 1
    local shape = self._lists_info[idx]["shape"]
    local str_padding = string.rep(' ', self._sizes_info.shape_list + offset - #shape)
    return ('%s, %s'):format(shape, str_padding)
end

function SetLoader:__len__()
    return self.nelems
end

function SetLoader:__tostring__()
    return ('SetLoader: set<%s>, len<%d>'):format(self.set, self.nelems)
end


---------------------------------------------------------------------------------------------------

function FieldLoader:__init(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Field metadata loader class.

            This class contains several methods to fetch data from a specific
            field of a set (group) in a hdf5 file. It contains useful information
            about the field and also several methods to fetch data.

            Parameters
            ----------
            hdf5_field : hdf5.HDF5DataSet
                hdf5 field object handler.
            obj_id : number
                Position of the field in 'object_fields'.

            Attributes
            ----------
            data : hdf5.HDF5DataSet
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
        ]],
        {name="hdf5_field", type="hdf5.HDF5DataSet",
         help="hdf5 field object handler."},
        {name="obj_id", type="number",
         help="Position of the field in 'object_fields'.",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    self.data = args.hdf5_field
    self.hdf5_handler = args.hdf5_field
    self._in_memory = false
    self.set = self:_get_set_name()
    self.name = self:_get_field_name()
    self._size = self:_get_field_size()
    self.shape = get_data_shape(self._size)
    self.type = get_data_type_hdf5(self.data, self._size)
    self.ids_list = self:_get_ids_list()
    self.ndims = #self._size
    -- fillvalue not implemented in hdf5 lib
    self.obj_id = args.obj_id
end

function FieldLoader:_get_set_name()
    local hdf5_object_str = self:_get_hdf5_object_str()
    local str = split_str(hdf5_object_str, '/')
    return str[1]
end

function FieldLoader:_get_field_name()
    local hdf5_object_str = self:_get_hdf5_object_str()
    local str = split_str(hdf5_object_str, '/')
    return str[2]
end

function FieldLoader:_get_hdf5_object_str()
    return hdf5._getObjectName(self.hdf5_handler._datasetID)
end

function FieldLoader:_get_field_size()
    return self.data:dataspaceSize()
end

function FieldLoader:_get_ids_list()
    local ids = {}
    for i=1, #self._size do
        table.insert(ids, {1, self._size[i]})
    end
    return ids
end

function FieldLoader:get(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves data of the field from the dataset's hdf5 metadata file.

            This method retrieves the i'th data from the hdf5 file. Also, it is
            possible to retrieve multiple values by inserting a list/tuple of
            number values as indexes.

            Parameters
            ----------
            index : number/table, optional
                Index number of the field. If it is a list, returns the data
                for all the value indexes of that list.

            Returns
            -------
            torch.*Tensor
                Array containing the field's data.
        ]],
        {name="index", type="table", default={},
         help="Index number of the field. If it is a list, returns the data " ..
              "for all the value indexes of that list.",
         opt=true}
    }

    -- Workaround to manage have multiple types for the same input.
    -- First the input checks if it is a number. If the input arg
    -- is not a number, do a second argument parsing to check if the
    -- second type matches the input argument.
    local initcheck_ = argcheck{
        quiet=true,
        pack=true,
        {name="index", type="number"}
    }

    local status, args = initcheck_(...)

    if status then
        return self:_get_range({{args.index, args.index}})
    else
        local args = initcheck(...)

        if next(args.index) then
            return self:_get_range({args.index})
        else
            return self:_get_all()
        end
    end
end

function FieldLoader:_get_range(idx)
    assert(idx)
    if self._in_memory then
        local data = self:_get_data_memory(idx)
        if type(data) ~= 'number' then
            data = data:squeeze(1)
        end
        return data
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
    local id = self:_get_ids(idx)
    return self.data:partial(unpack(id)):squeeze()
end

function FieldLoader:_get_ids(idx)
    assert(idx)
    local dtype = type(idx)
    if dtype == 'number' then
        return self:_get_ids_table({idx})
    elseif dtype == 'table' then
        if next(idx) then
            return self:_get_ids_table(idx)
        else
            return self.ids_list
        end
    else
        error('Invalid index type: ' .. dtype)
    end
end

function FieldLoader:_get_ids_table(idx)
    local ids = self.ids_list
    for i=1, #idx do
        local dtype = type(idx[i])
        if dtype == 'table' then
            ids[i][1] = idx[i][1] or ids[i][1]
            ids[i][2] = idx[i][2] or ids[i][2]
        elseif dtype == 'number' then
            ids[i][1] = idx[i]
            ids[i][2] = idx[i]
        else
            error('Invalid index type: ' .. dtype)
        end
    end
    return ids
end

function FieldLoader:_get_all()
    if self._in_memory then
        return self.data
    else
        return self.data:all()
    end
end

function FieldLoader:size(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Size of the field.

            Returns the number of the elements of the field.

            Returns
            -------
            table
                Returns the size of the field.
        ]]
    }

    local args = initcheck(...)

    return self._size
end

function FieldLoader:object_field_id(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Retrieves the index position of the field in the 'object_ids' list.

            This method returns the position of the field in the 'object_ids' object.
            If the field is not contained in this object, it returns a null value.

            Returns
            -------
            number
                Index of the field in the 'object_ids' list.
        ]]
    }

    local args = initcheck(...)

    return self.obj_id
end

function FieldLoader:info(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Prints information about the field.

            Displays information like name, size and shape of the field.

            Parameters
            ----------
            verbose : bool, optional
                If true, display extra information about the field.
        ]],
        {name="verbose", type="boolean", default=true,
         help="Name of the dataset.",
         opt=true}
    }

    local args = initcheck(...)

    if args.verbose then
        if self.obj_id then
            print(('Field: %s,  shape = %s,  dtype = %s,  (in \'object_ids\', position = %d)')
            :format(self.name, self.shape, self.type, self.obj_id))
        else
            print(('Field: %s,  shape = %s,  dtype = %s')
            :format(self.name, self.shape, self.type))
        end
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

function FieldLoader:to_memory(...)
    local initcheck = argcheck{
        pack=true,
        help=[[
            Modifies how data is accessed and stored.

            Accessing data from a field can be done in two ways: memory or disk.
            To enable data allocation and access from memory requires the user to
            specify a boolean. If set to True, data is allocated to a numpy ndarray
            and all accesses are done in memory. Otherwise, data is kept in disk and
            accesses are done using the HDF5 object handler.

            Parameters
            ----------
            in_memory : bool
                Move the data to memory (if true).
                Otherwise, move to disk (hdf5).
        ]],
        {name="in_memory", type="boolean", default=true,
         help="Move the data to memory (if True). Otherwise, move to disk (hdf5).",
         opt = true}
    }

    -- parse options
    local args = initcheck(...)

    self:_set_to_memory(args.in_memory)
end

function FieldLoader:__len__()
    return self._size
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
