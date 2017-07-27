--[[
    Padding functions.
--]]


------------------------------------------------------------------------------------------------------------

local function convert_to_table_of_tables(inputA)
    assert(inputA)
    local out = {}
    for _, v in ipairs(inputA) do
        table.insert(out, v:totable())
    end
    return out
end

------------------------------------------------------------------------------------------------------------

local function pad_table(tableA, val)
    assert(tableA)
    assert(val)

    -- get maximum size of all tables
    local max_lenght = 0
    for _, v in pairs(tableA) do
        max_lenght = math.max(max_lenght, #v)
    end

    -- pad table with 'val'
    local out = {}
    for _, v in ipairs(tableA) do
        -- copy contents into a new table
        local t = {}
        for _, value in ipairs(v) do
            table.insert(t, value)
        end
        -- pad table
        for i=1, max_lenght - #v do
            table.insert(t, val)
        end
        -- add padded table
        table.insert(out,t)
    end

    if #out > 1 then
        return out
    else
        return out[1]
    end
end

------------------------------------------------------------------------------------------------------------

--[[ pad a table of tables or a table of tensors into a table of tables with a value ]]--
local function pad_list(inputA, val)
    assert(inputA)

    local val = val or -1

    if type(inputA) == 'userdata' then
        return pad_table(inputA:totable(), val)
    elseif type(inputA) == 'table' then
        local tmp = inputA
        if type(inputA[1]) == 'userdata' then
            tmp = convert_to_table_of_tables(inputA)
        end
        return pad_table(tmp, val)
    else
        error(('Invalid input type for pad_list: \'%s\'. Must be either a table or a tensor.')
              :format(type(inputA)))
    end
end

------------------------------------------------------------------------------------------------------------

local function unpad_table(tableA, val)
    assert(tableA)
    assert(val)

    local out = {}
    if type(tableA[1]) == 'table' then
        for _, v in ipairs(tableA) do
            local t = {}
            for _, value in ipairs(v) do
                if value ~= val then
                    table.insert(t, value)
                end
            end
            table.insert(out, t)
        end
    else
        for _, v in ipairs(tableA) do
            if v ~= val then
                table.insert(out, v)
            end
        end
    end

    if #out > 1 then
        return out
    else
        return out[1]
    end
end

------------------------------------------------------------------------------------------------------------

--[[ unpad a table of tables or table of tensors with a
     certain value into a table of tables. ]]--
local function unpad_list(inputA, val)
    assert(inputA)

    local val = val or -1

    if type(inputA) == 'userdata' then
        return unpad_table(inputA:totable(), val)
    elseif type(inputA) == 'table' then
        local tmp = inputA
        if type(inputA[1]) == 'userdata' then
            tmp = convert_to_table_of_tables(inputA)
        end
        return unpad_table(tmp, val)
    else
        error(('Invalid input type for unpad_list: \'%s\'. Must be either a table or a tensor.')
              :format(type(inputA)))
    end
end

------------------------------------------------------------------------------------------------------------

return {
    pad_list = pad_list,
    unpad_list = unpad_list,
}