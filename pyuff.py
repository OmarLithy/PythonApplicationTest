
import os
import numpy as np
import struct
import warnings
import sys
warnings.simplefilter("default")

def _opt_fields(dict, fields_dict):
    """Sets the optional fields of the dict dictionary. 
    
    Optionally fields are given in fields_dict dictionary.
    
    :param dict: dictionary to be updated
    :param fields_dict: dictionary with default values
    """

    for key in fields_dict:
        #             if not dict.has_key(key):
        if not key in dict:
            dict.update({key: fields_dict[key]})
    return dict


def _parse_header_line(line, min_values, widths, types, names):
    """Parses the given line (a record in terms of UFF file) and returns all
    the fields. 

    Fields are split according to their widths as given in widths.
    min_values specifies how many values (fields) are mandatory to read
    from the line. Also, number of values found must not exceed the
    number of fields requested by fieldsIn.
    On the output, a dictionary of field names and corresponding
    filed values is returned.
    
    :param line: a string representing the whole line.
    :param min_values: specifies how many values (fields) are mandatory to read
        from the line
    :param widths: fields widths to be read
    :param types: field types 1=string, 2=int, 3=float, -1=ignore the field
    :param names: a list of key (field) names
    """
    fields = {}
    n_fields_req = len(names)
    fields_from_line = []
    fields_out = {}

    # Extend the line if shorter than 80 chars
    ll = len(line)
    if ll < 80:
        line = line + ' ' * (80 - ll)
    # Parse the line for fields
    si = 0
    for n in range(0, len(widths)):
        fields_from_line.append(line[si:si + widths[n]].strip())
        si += widths[n]
    # Check for the number of fields,...
    n_fields = len(fields_from_line)
    if (n_fields_req < n_fields) or (min_values > n_fields):
        raise Exception('Error parsing header section; too many or to less' + \
                            'fields found')
    # Mandatory fields
    for key, n in zip(names[:min_values], range(0, min_values)):
        if types[n] == -1:
            pass
        elif types[n] == 1:
            fields_out.update({key: fields_from_line[n]})
        elif types[n] == 2:
            fields_out.update({key: int(fields_from_line[n])})
        else:
            fields_out.update({key: float(fields_from_line[n])})
    # Optional fields
    for key, n in zip(names[min_values:n_fields], range(min_values, n_fields)):
        try:
            if types[n] == -1:
                pass
            elif types[n] == 1:
                fields_out.update({key: fields_from_line[n]})
            elif types[n] == 2:
                fields_out.update({key: int(fields_from_line[n])})
            else:
                fields_out.update({key: float(fields_from_line[n])})
        except ValueError:
            if types[n] == 1:
                fields_out.update({key: ''})
            elif types[n] == 2:
                fields_out.update({key: 0})
            else:
                fields_out.update({key: 0.0})
    return fields_out

def check_dict_for_none(dataset):
    dataset1 = {}
    for k, v in dataset.items():
        if v is not None:
            dataset1[k] = v

    return dataset1

def get_structure_58(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """ """
    if raw:
        return out
    else:
        print(out)   

def _extract58(block_data):
    """Extract function at nodal DOF - data-set 58."""
    dset = {'type': 58, 'binary': 0}
    try:
        binary = False
        split_header = b''.join(block_data.splitlines(True)[:13]).decode('utf-8',  errors='replace').splitlines(True)
        if len(split_header[1]) >= 7:
            if split_header[1][6].lower() == 'b':
                # Read some addititional fields from the header section
                binary = True
                dset['binary'] = 1
                dset.update(_parse_header_line(split_header[1], 6, [6, 1, 6, 6, 12, 12, 6, 6, 12, 12],
                                                    [-1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
                                                    ['', '', 'byte_ordering', 'fp_format', 'n_ascii_lines',
                                                        'n_bytes', '', '', '', '']))
        dset.update(_parse_header_line(split_header[2], 1, [80], [1], ['id1']))
        dset.update(_parse_header_line(split_header[3], 1, [80], [1], ['id2']))
        dset.update(_parse_header_line(split_header[4], 1, [80], [1], ['id3']))  # usually for the date
        dset.update(_parse_header_line(split_header[5], 1, [80], [1], ['id4']))
        dset.update(_parse_header_line(split_header[6], 1, [80], [1], ['id5']))
        dset.update(_parse_header_line(split_header[7], 1, [5, 10, 5, 10, 11, 10, 4, 11, 10, 4],
                                            [2, 2, 2, 2, 1, 2, 2, 1, 2, 2],
                                            ['func_type', 'func_id', 'ver_num', 'load_case_id', 'rsp_ent_name',
                                                'rsp_node', 'rsp_dir', 'ref_ent_name',
                                                'ref_node', 'ref_dir']))
        dset.update(_parse_header_line(split_header[8], 6, [10, 10, 10, 13, 13, 13], [2, 2, 2, 3, 3, 3],
                                            ['ord_data_type', 'num_pts', 'abscissa_spacing', 'abscissa_min',
                                                'abscissa_inc', 'z_axis_value']))
        dset.update(_parse_header_line(split_header[9], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['abscissa_spec_data_type', 'abscissa_len_unit_exp',
                                                'abscissa_force_unit_exp', 'abscissa_temp_unit_exp',
                                                'abscissa_axis_lab', 'abscissa_axis_units_lab']))
        dset.update(_parse_header_line(split_header[10], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['ordinate_spec_data_type', 'ordinate_len_unit_exp',
                                                'ordinate_force_unit_exp', 'ordinate_temp_unit_exp',
                                                'ordinate_axis_lab', 'ordinate_axis_units_lab']))
        dset.update(_parse_header_line(split_header[11], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['orddenom_spec_data_type', 'orddenom_len_unit_exp',
                                                'orddenom_force_unit_exp', 'orddenom_temp_unit_exp',
                                                'orddenom_axis_lab', 'orddenom_axis_units_lab']))
        dset.update(_parse_header_line(split_header[12], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['z_axis_spec_data_type', 'z_axis_len_unit_exp',
                                                'z_axis_force_unit_exp', 'z_axis_temp_unit_exp', 'z_axis_axis_lab',
                                                'z_axis_axis_units_lab']))
        # Body
        # split_data = ''.join(split_data[13:])
        if binary:
            try:     
                split_data = b''.join(block_data.splitlines(True)[13:])
                if dset['byte_ordering'] == 1:
                    bo = '<'
                else:
                    bo = '>'
                if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 5):
                    # single precision - 4 bytes
                    values = np.asarray(struct.unpack('%c%sf' % (bo, int(len(split_data) / 4)), split_data), 'd')
                else:
                    # double precision - 8 bytes
                    values = np.asarray(struct.unpack('%c%sd' % (bo, int(len(split_data) / 8)), split_data), 'd')
            except:
                raise Exception('Potentially wrong data format (common with binary files from some commercial softwares). Try using pyuff.fix_58b() to fix your file. For more information, see https://github.com/ladisk/pyuff/issues/61')
        else:
            values = []
            split_data = block_data.decode('utf-8', errors='replace').splitlines(True)[13:]
            if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 5):
                for line in split_data[:-1]:  # '6E13.5'
                    values.extend([float(line[13 * i:13 * (i + 1)]) for i in range(len(line) // 13)])
                else:
                    line = split_data[-1]
                    values.extend([float(line[13 * i:13 * (i + 1)]) for i in range(len(line) // 13) if line[13 * i:13 * (i + 1)]!='             '])
            elif ((dset['ord_data_type'] == 4) or (dset['ord_data_type'] == 6)) and (dset['abscissa_spacing'] == 1):
                for line in split_data:  # '4E20.12'
                    values.extend([float(line[20 * i:20 * (i + 1)]) for i in range(len(line) // 20)])
            elif (dset['ord_data_type'] == 4) and (dset['abscissa_spacing'] == 0):
                for line in split_data:  # 2(E13.5,E20.12)
                    values.extend(
                        [float(line[13 * (i + j) + 20 * (i):13 * (i + 1) + 20 * (i + j)]) \
                            for i in range(len(line) // 33) for j in [0, 1]])
            elif (dset['ord_data_type'] == 6) and (dset['abscissa_spacing'] == 0):
                for line in split_data:  # 1E13.5,2E20.12
                    values.extend([float(line[0:13]), float(line[13:33]), float(line[33:53])])
            else:
                raise Exception('Error reading data-set #58b; not proper data case.')

            values = np.asarray(values)
            # values = np.asarray([float(str) for str in split_data],'d')
        if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 4):
            # Non-complex ordinate data
            if (dset['abscissa_spacing'] == 0):
                # Uneven abscissa
                dset['x'] = values[:-1:2].copy()
                dset['data'] = values[1::2].copy()
            else:
                # Even abscissa
                n_val = len(values)
                min_val = dset['abscissa_min']
                d = dset['abscissa_inc']
                dset['x'] = min_val + np.arange(n_val) * d
                dset['data'] = values.copy()
        elif (dset['ord_data_type'] == 5) or (dset['ord_data_type'] == 6):
            # Complex ordinate data
            if (dset['abscissa_spacing'] == 0):
                # Uneven abscissa
                dset['x'] = values[:-2:3].copy()
                dset['data'] = values[1:-1:3] + 1.j * values[2::3]
            else:
                # Even abscissa
                n_val = len(values) / 2
                min_val = dset['abscissa_min']
                d = dset['abscissa_inc']
                dset['x'] = min_val + np.arange(n_val) * d
                dset['data'] = values[0:-1:2] + 1.j * values[1::2]
        del values
    except:
        raise Exception('Error reading data-set #58b')
    return dset


def prepare_58(
        binary=None,
        id1=None,
        id2=None,
        id3=None,
        id4=None,
        id5=None,

        func_type=None,
        ver_num=None,
        load_case_id=None,
        rsp_ent_name=None,
        rsp_node=None,
        rsp_dir=None,
        ref_ent_name=None,
        ref_node=None,
        ref_dir=None,

        ord_data_type=None,
        num_pts=None,
        abscissa_spacing=None,
        abscissa_min=None,
        abscissa_inc=None,
        z_axis_value=None,

        abscissa_spec_data_type=None,
        abscissa_len_unit_exp=None,
        abscissa_force_unit_exp=None,
        abscissa_temp_unit_exp=None,
        
        abscissa_axis_units_lab=None,

        ordinate_spec_data_type=None,
        ordinate_len_unit_exp=None,
        ordinate_force_unit_exp=None,
        ordinate_temp_unit_exp=None,
        
        ordinate_axis_units_lab=None,

        orddenom_spec_data_type=None,
        orddenom_len_unit_exp=None,
        orddenom_force_unit_exp=None,
        orddenom_temp_unit_exp=None,
        
        orddenom_axis_units_lab=None,

        z_axis_spec_data_type=None,
        z_axis_len_unit_exp=None,
        z_axis_force_unit_exp=None,
        z_axis_temp_unit_exp=None,
        
        z_axis_axis_units_lab=None,

        data=None,
        x=None,
        spec_data_type=None,
        byte_ordering=None,
        fp_format=None,
        n_ascii_lines=None,
        n_bytes=None,
        return_full_dict=False):

    """Name:   Function at Nodal DOF

    R-Record, F-Field

    :param binary: 1 for binary, 0 for ascii, optional
    :param id1: R1 F1, ID Line 1, optional
    :param id2: R2 F1, ID Line 2, optional
    :param id3: R3 F1, ID Line 3, optional
    :param id4: R4 F1, ID Line 4, optional
    :param id5: R5 F1, ID Line 5, optional

    **DOF identification**

    :param func_type: R6 F1, Function type
    :param ver_num: R6 F3, Version number, optional
    :param load_case_id: R6 F4, Load case identification number, optional
    :param rsp_ent_name: R6 F5, Response entity name, optional
    :param rsp_node: R6 F6, Response node
    :param rsp_dir: R6 F7, Responde direction
    :param ref_ent_name: R6 F8, Reference entity name, optional
    :param ref_node: R6 F9, Reference node
    :param ref_dir: R6 F10, Reference direction

    **Data form**

    :param ord_data_type: R7 F1, Ordinate data type, ignored
    :param num_pts: R7 F2, number of data pairs for uneven abscissa or number of data values for even abscissa, ignored
    :param abscissa_spacing: R7 F3, Abscissa spacing (0- uneven, 1-even), ignored
    :param abscissa_min: R7 F4, Abscissa minimum (0.0 if spacing uneven), ignored
    :param abscissa_inc: R7 F5, Abscissa increment (0.0 if spacing uneven), ignored
    :param z_axis_value: R7 F6, Z-axis value (0.0 if unused), optional

    **Abscissa data characteristics**

    :param abscissa_spec_data_type: R8 F1, Abscissa specific data type, optional
    :param abscissa_len_unit_exp: R8 F2, Abscissa length units exponent, optional
    :param abscissa_force_unit_exp: R8 F3, Abscissa force units exponent, optional
    :param abscissa_temp_unit_exp: R8 F4, Abscissa temperature units exponent, optional
    
    :param abscissa_axis_units_lab: R8 F6, Abscissa units label, optional

    **Ordinate (or ordinate numerator) data characteristics**

    :param ordinate_spec_data_type: R9 F1, Ordinate specific data type, optional
    :param ordinate_len_unit_exp: R9 F2, Ordinate length units exponent, optional
    :param ordinate_force_unit_exp: R9 F3, Ordinate force units exponent, optional
    :param ordinate_temp_unit_exp: R9 F4, Ordinate temperature units exponent, optional
    
    :param ordinate_axis_units_lab: R9 F6, Ordinate units label, optional

    **Ordinate denominator data characteristics**

    :param orddenom_spec_data_type: R10 F1, Ordinate Denominator specific data type, optional
    :param orddenom_len_unit_exp: R10 F2, Ordinate Denominator length units exponent, optional
    :param orddenom_force_unit_exp: R10 F3, Ordinate Denominator force units exponent, optional
    :param orddenom_temp_unit_exp: R10 F4, Ordinate Denominator temperature units exponent, optional
    
    :param orddenom_axis_units_lab: R10 F6, Ordinate Denominator units label, optional

    **Z-axis data characteristics**

    :param z_axis_spec_data_type:  R11 F1, Z-axis specific data type, optional
    :param z_axis_len_unit_exp: R11 F2, Z-axis length units exponent, optional
    :param z_axis_force_unit_exp: R11 F3, Z-axis force units exponent, optional
    :param z_axis_temp_unit_exp: R11 F4, Z-axis temperature units exponent, optional
    
    :param z_axis_axis_units_lab: R11 F6, Z-axis units label, optional

    **Data values**

    :param data: R12 F1, Data values

    :param x: Abscissa array
    :param spec_data_type: Specific data type, optional
    :param byte_ordering: R1 F3, Byte ordering (only for binary), ignored
    :param fp_format: R1 F4 Floating-point format (only for binary), ignored
    :param n_ascii_lines: R1 F5, Number of ascii lines (only for binary), ignored
    :param n_bytes: R1 F6, Number of bytes (only for binary), ignored

    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_58**

    >>> save_to_file = 'test_pyuff'
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>> uff_datasets = []
    >>> binary = [0, 1, 0]  # ascii of binary
    >>> frequency = np.arange(10)
    >>> np.random.seed(0)
    >>> for i, b in enumerate(binary):
    >>>     print('Adding point {}'.format(i + 1))
    >>>     response_node = 1
    >>>     response_direction = 1
    >>>     reference_node = i + 1
    >>>     reference_direction = 1
    >>>     # this is an artificial 'frf'
    >>>     acceleration_complex = np.random.normal(size=len(frequency)) + 1j * np.random.normal(size=len(frequency))
    >>>     name = 'TestCase'
    >>>     data = pyuff.prepare_58(
    >>>         binary=binary[i],
    >>>         func_type=4,
    >>>         rsp_node=response_node,
    >>>         rsp_dir=response_direction,
    >>>         ref_dir=reference_direction,
    >>>         ref_node=reference_node,
    >>>         data=acceleration_complex,
    >>>         x=frequency,
    >>>         id1='id1',
    >>>         rsp_ent_name=name,
    >>>         ref_ent_name=name,
    >>>         abscissa_spacing=1,
    >>>         abscissa_spec_data_type=18,
    >>>         ordinate_spec_data_type=12,
    >>>         orddenom_spec_data_type=13)
    >>>     uff_datasets.append(data.copy())
    >>>     if save_to_file:
    >>>         uffwrite = pyuff.UFF(save_to_file)
    >>>         uffwrite._write_set(data, 'add')
    >>> uff_datasets
    """

    if binary not in (0, 1, None):
        raise ValueError('binary can be 0 or 1')
    if type(id1) != str and id1 != None:
        raise TypeError('id1 must be string.')
    if type(id2) != str and id2 != None:
        raise TypeError('id2 must be string.')
    if type(id3) != str and id3 != None:
        raise TypeError('id3 must be string.')
    if type(id4) != str and id4 != None:
        raise TypeError('id4 must be string.')
    if type(id5) != str and id5 != None:
        raise TypeError('id5 must be string.')
    
    if func_type not in np.arange(28) and func_type != None:
        raise ValueError('func_type must be integer between 0 and 27')
    if np.array(ver_num).dtype != int and ver_num != None:
        raise TypeError('ver_num must be integer')
    if np.array(load_case_id).dtype != int and load_case_id != None:
        raise TypeError('load_case_id must be integer')
    if type(rsp_ent_name) != str and rsp_ent_name != None:
        raise TypeError('rsp_ent_name must be string')
    if np.array(rsp_node).dtype != int and rsp_node != None:
        raise TypeError('rsp_node must be integer')
    if rsp_dir not in np.arange(-6,7) and rsp_dir != None:
        raise ValueError('rsp_dir must be integer between -6 and 6')
    if type(ref_ent_name) != str and ref_ent_name != None:
        raise TypeError('rsp_ent_name must be string')
    if np.array(ref_node).dtype != int and ref_node != None:
        raise TypeError('ref_node must be int')
    if ref_dir not in np.arange(-6,7) and ref_dir != None:
        raise ValueError('ref_dir must be integer between -6 and 6')
    
    if ord_data_type not in (2, 4, 5, 6, None):
        raise ValueError('ord_data_type can be: 2,4,5,6')
    if np.array(num_pts).dtype != int and num_pts != None:
        raise TypeError('num_pts must be integer')
    if abscissa_spacing not in (0, 1, None):
        raise ValueError('abscissa_spacing can be 0:uneven, 1:even')
    if np.array(abscissa_min).dtype != float and abscissa_min != None:
        raise TypeError('abscissa_min must be float')
    if np.array(abscissa_inc).dtype != float and abscissa_inc != None:
        raise TypeError('abscissa_inc must be float')
    if np.array(z_axis_value).dtype != float and z_axis_value != None:
        raise TypeError('z_axis_value must be float')
    
    if abscissa_spec_data_type not in np.arange(21) and abscissa_spec_data_type != None:
        raise ValueError('abscissa_spec_data_type must be integer between 0 nd 21')
    if np.array(abscissa_len_unit_exp).dtype != int and abscissa_len_unit_exp != None:
        raise TypeError('abscissa_len_unit_exp must be integer')
    if np.array(abscissa_force_unit_exp).dtype != int and abscissa_force_unit_exp != None:
        raise TypeError('abscissa_force_unit_exp must be integer')
    if np.array(abscissa_temp_unit_exp).dtype != int and abscissa_temp_unit_exp != None:
        raise TypeError('abscissa_temp_unit_exp must be integer')
    if type(abscissa_axis_units_lab) != str and abscissa_axis_units_lab != None:
        raise TypeError('abscissa_axis_units_lab must be string')

    if ordinate_spec_data_type not in np.arange(21) and ordinate_spec_data_type != None:
        raise ValueError('ordinate_spec_data_type must be integer between 0 nd 21')
    if np.array(ordinate_len_unit_exp).dtype != int and ordinate_len_unit_exp != None:
        raise TypeError('ordinate_len_unit_exp must be integer')
    if np.array(ordinate_force_unit_exp).dtype != int and ordinate_force_unit_exp != None:
        raise TypeError('ordinate_force_unit_exp must be integer')
    if np.array(ordinate_temp_unit_exp).dtype != int and ordinate_temp_unit_exp != None:
        raise TypeError('ordinate_temp_unit_exp must be integer')
    if type(ordinate_axis_units_lab) != str and ordinate_axis_units_lab != None:
        raise TypeError('ordinate_axis_units_lab must be string')

    if orddenom_spec_data_type not in np.arange(21) and orddenom_spec_data_type != None:
        raise ValueError('orddenom_spec_data_type must be integer between 0 nd 21')
    if np.array(orddenom_len_unit_exp).dtype != int and orddenom_len_unit_exp != None:
        raise TypeError('orddenom_len_unit_exp must be integer')
    if np.array(orddenom_force_unit_exp).dtype != int and orddenom_force_unit_exp != None:
        raise TypeError('orddenom_force_unit_exp must be integer')
    if np.array(orddenom_temp_unit_exp).dtype != int and orddenom_temp_unit_exp != None:
        raise TypeError('orddenom_temp_unit_exp must be integer')
    if type(orddenom_axis_units_lab) != str and orddenom_axis_units_lab != None:
        raise TypeError('orddenom_axis_units_lab must be string')

    if z_axis_spec_data_type not in np.arange(21) and z_axis_spec_data_type != None:
        raise ValueError('z_axis_spec_data_type must be integer between 0 nd 21')
    if np.array(z_axis_len_unit_exp).dtype != int and z_axis_len_unit_exp != None:
        raise TypeError('z_axis_len_unit_exp must be integer')
    if np.array(z_axis_force_unit_exp).dtype != int and z_axis_force_unit_exp != None:
        raise TypeError('z_axis_force_unit_exp must be integer')
    if np.array(z_axis_temp_unit_exp).dtype != int and z_axis_temp_unit_exp != None:
        raise TypeError('z_axis_temp_unit_exp must be integer')
    if type(z_axis_axis_units_lab) != str and z_axis_axis_units_lab != None:
        raise TypeError('z_axis_axis_units_lab must be string')
    
    if np.array(data).dtype != float and np.array(data).dtype != complex:
        if data != None:
            raise TypeError('data must be float')
    


    dataset={
        'type': 58,
        'binary': binary,
        'id1': id1,
        'id2': id2,
        'id3': id3,
        'id4': id4,
        'id5': id5,

        'func_type': func_type,
        'ver_num': ver_num,
        'load_case_id': load_case_id,
        'rsp_ent_name': rsp_ent_name,
        'rsp_node': rsp_node,
        'rsp_dir': rsp_dir,
        'ref_ent_name': ref_ent_name,
        'ref_node': ref_node,
        'ref_dir': ref_dir,

        'ord_data_type': ord_data_type,
        'num_pts': num_pts,
        'abscissa_spacing': abscissa_spacing,
        'abscissa_min': abscissa_min,
        'abscissa_inc': abscissa_inc,
        'z_axis_value': z_axis_value,

        'abscissa_spec_data_type': abscissa_spec_data_type,
        'abscissa_len_unit_exp': abscissa_len_unit_exp,
        'abscissa_force_unit_exp': abscissa_force_unit_exp,
        'abscissa_temp_unit_exp': abscissa_temp_unit_exp,
        
        'abscissa_axis_units_lab': abscissa_axis_units_lab,

        'ordinate_spec_data_type': ordinate_spec_data_type,
        'ordinate_len_unit_exp': ordinate_len_unit_exp,
        'ordinate_force_unit_exp': ordinate_force_unit_exp,
        'ordinate_temp_unit_exp': ordinate_temp_unit_exp,
        
        'ordinate_axis_units_lab': ordinate_axis_units_lab,

        'orddenom_spec_data_type': orddenom_spec_data_type,
        'orddenom_len_unit_exp': orddenom_len_unit_exp,
        'orddenom_force_unit_exp': orddenom_force_unit_exp,
        'orddenom_temp_unit_exp': orddenom_temp_unit_exp,
        
        'orddenom_axis_units_lab': orddenom_axis_units_lab,

        'z_axis_spec_data_type': z_axis_spec_data_type,
        'z_axis_len_unit_exp': z_axis_len_unit_exp,
        'z_axis_force_unit_exp': z_axis_force_unit_exp,
        'z_axis_temp_unit_exp': z_axis_temp_unit_exp,
        
        'z_axis_axis_units_lab': z_axis_axis_units_lab,

        'data': data,
        'x': x,
        'spec_data_type': spec_data_type,
        'byte_ordering': byte_ordering,
        'fp_format': fp_format,
        'n_ascii_lines': n_ascii_lines,
        'n_bytes': n_bytes
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)


    return dataset

_SUPPORTED_SETS = ['58']


class UFF:
    """
    Manages data reading and writing from/to the UFF file.
    
    The UFF class instance requires exactly 1 parameter - a file name of a
    universal file. If the file does not exist, no basic file info will be
    extracted and the status will be False - indicating that the file is not
    refreshed. Hovewer, when one tries to read one or more data-sets, the file
    must exist or the Exception will be raised.
    
    The file, given as a parameter to the UFF instance, is open only when
    reading from or writing to the file. The UFF instance refreshes the file
    automatically - use ``UFF.get_status()`` to see the refresh status); note
    that this works fine if the file is being changed only through the UFF
    instance and not by other functions or even by other means, e.g.,
    externally. If the file is changed externally, the ``UFF.refresh()`` should
    be invoked before any reading or writing.
    
    All array-type data are read/written using numpy's ``np.array`` module.
    """

    def __init__(self, filename=None, fileName=None):
        """
        Initializes the uff object and extract the basic info: 
        the number of sets, types of the sets and format of the sets (ascii
        or binary). To manually refresh this info, call the refresh method
        manually.
        
        Whenever some data is written to a file, a read-only flag 
        indicates that the file needs to be refreshed - before any reading,
        the file is refreshed automatically (when needed).
        """
        # Some "private" members
        if filename != None:
            self._filename = filename
        
        elif fileName != None:
            self._filename = fileName
            warnings.warn('Argument ``fileName`` will be deprecated in the future. Please use ``filename``')
        

        
        self._block_ind = []  # an array of block indices: start-end pairs in rows
        self._refreshed = False
        self._n_sets = 0  # number of sets found in file
        self._set_types = np.array(())  # list of set-type numbers
        self._set_formats = np.array(())  # list of set-format numbers (0=ascii,1=binary)
        # Refresh
        self.refresh()

    def get_supported_sets(self):
        """Returns a list of data-sets supported for reading and writing."""
        return _SUPPORTED_SETS

    def get_n_sets(self):
        """
        Returns the number of valid sets found in the file."""
        if not self._refreshed:
            self.refresh()
        return self._n_sets

    def get_set_types(self):
        """
        Returns an array of data-set types. All valid data-sets are returned,
        even those that are not supported, i.e., whose contents will not be
        read.
        """
        if not self._refreshed:
            self.refresh()
        return self._set_types

    def get_set_formats(self):
        """Returns an array of data-set formats: 0=ascii, 1=binary."""
        if not self._refreshed:
            self.refresh()
        return self._set_formats

    def get_file_name(self):
        """Returns the file name (as a string) associated with the uff object."""
        return self._filename

    def file_exists(self):
        """
        Returns true if the file exists and False otherwise. If the file does
        not exist, invoking one of the read methods would raise the Exception
        exception.
        """
        return os.path.exists(self._filename)

    def get_status(self):
        """
        Returns the file status, i.e., True if the file is refreshed and
        False otherwise.
        """
        return self._refreshed

    def refresh(self):
        """
        Extract/refreshes the info of all the sets from UFF file (if the file
        exists). The file must exist and must be accessable otherwise, an
        error is raised. If the file cannot be refreshed, False is returned and
        True otherwise.
        """
        self._refreshed = False
        if not self.file_exists():
            return False  # cannot read the file if it does not exist
        try:
            fh = open(self._filename, 'rb')
        #             fh = open(self._filename, 'rt')
        except:
            raise Exception('Cannot access the file %s' % self._filename)
        else:
            try:
                # Parses the entire file for '    -1' tags and extracts
                # the corresponding indices
                data = fh.read()
                data_len = len(data)
                ind = -1
                block_ind = []
                while True:
                    ind = data.find(b'    -1', ind + 1)
                    if ind == -1:
                        break
                    block_ind.append(ind)
                block_ind = np.asarray(block_ind, dtype='int64')

                # Constructs block indices of start and end values; each pair
                # points to start and end offset of the data-set (block) data,
                # but, the start '    -1' tag is included while the end one is
                # excluded.
                n_blocks = int(np.floor(len(block_ind) / 2.0))
                if n_blocks == 0:
                    # No valid blocks found but the file is still considered
                    # being refreshed
                    fh.close()
                    self._refreshed = True
                    return self._refreshed
                self._block_ind = np.zeros((n_blocks, 2), dtype='int64')
                self._block_ind[:, 0] = block_ind[:-1:2].copy()
                self._block_ind[:, 1] = block_ind[1::2].copy() - 1

                # Go through all the data-sets (blocks) and extract data-set
                # type and the property whether the data-set is in binary
                # or ascii format
                self._n_sets = n_blocks
                self._set_types = np.zeros(n_blocks).astype(int)
                self._set_formats = np.zeros(n_blocks)
                for ii in range(0, self._n_sets):
                    si = self._block_ind[ii, 0]
                    ei = self._block_ind[ii, 1]
                    try:
                        block_data = data[si:ei + 1].splitlines()
                        self._set_types[ii] = int(block_data[1][0:6])
                        if block_data[1][6].lower() == 'b':
                            self._set_formats[ii] = 1
                    except:
                        # Some non-valid blocks found; ignore the exception
                        pass
                del block_ind
            except:
                fh.close()
                raise Exception('Error refreshing UFF file: ' + self._filename)
            else:
                self._refreshed = True
                fh.close()
                return self._refreshed

    def read_sets(self, setn=None):
        """
        Reads sets from the list or array ``setn``. If ``setn=None``, all
        sets are read (default). Sets are numbered starting at 0, ending at
        n-1. The method returns a list of dset dictionaries - as
        many dictionaries as there are sets. Unknown data-sets are returned
        empty.
        
        User must be sure that, since the last reading/writing/refreshing,
        the data has not changed by some other means than through the
        UFF object.
        """
        dset = []
        if setn is None:
            read_range = range(0, self._n_sets)
        else:
            if (not type(setn).__name__ == 'list'):
                read_range = [setn]
            else:
                read_range = setn
        if not self.file_exists():
            raise Exception('Cannot read from a non-existing file: ' + self._filename)
        if not self._refreshed:
            if not self.refresh():
                raise Exception('Cannot read from the file: ' + self._filename)
        try:
            for ii in read_range:
                dset.append(self._read_set(ii))
        except Exception as msg:
            if hasattr(msg, 'value'):
                raise Exception('Error when reading ' + str(ii) + '-th data-set: ' + msg.value)
            else:
                raise Exception('Error when reading data-set(s).')
        if len(dset) == 1:
            dset = dset[0]
        return dset

    def _read_set(self, n):
        """
        Reads n-th set from UFF file. 
        n can be an integer between 0 and n_sets-1. 
        User must be sure that, since the last reading/writing/refreshing, 
        the data has not changed by some other means than through the
        UFF object. The method returns dset dictionary.
        """
        
        dset = {}
        if not self.file_exists():
            raise Exception('Cannot read from a non-existing file: ' + self._filename)
        if not self._refreshed:
            if not self.refresh():
                raise Exception('Cannot read from the file: ' + self._filename + '. The file cannot be refreshed.')
        if (n > self._n_sets - 1) or (n < 0):
            raise Exception('Cannot read data-set: ' + str(int(n)) + \
                               '. Data-set number to high or to low.')
        # Read n-th data-set data (one block)
        try:
            fh = open(self._filename, 'rb')
        except:
            raise Exception('Cannot access the file: ' + self._filename + ' to read from.')
        else:
            try:
                si = self._block_ind[n][0]  # start offset
                ei = self._block_ind[n][1]  # end offset
                fh.seek(si)
                if self._set_types[int(n)] == 58:
                    block_data = fh.read(ei - si + 1)  # decoding is handled later in _extract58
                else:
                    block_data = fh.read(ei - si + 1).decode('utf-8', errors='replace')
            except:
                fh.close()
                raise Exception('Error reading data-set #: ' + int(n))
            else:
                fh.close()
        # Extracts the dset
        if self._set_types[int(n)] == 58:
            dset = _extract58(block_data)
        else:
            dset['type'] = self._set_types[int(n)]
            # Unsupported data-set - do nothing
            pass
        return dset

    def _write_set(self, dset, mode='add', force_double=True):
        """
        Writes UFF data (UFF data-sets) to the file.  The mode can be
        either 'add' (default) or 'overwrite'. The dset is a
        dictionary of keys and corresponding values. Unsupported
        data-set will be ignored.
         
        For each data-set, there are some optional and some required fields at
        dset dictionary. Also, in general, the sum of the required
        and the optional fields together can be less then the number of fields
        read from the same type of data-set; the reason is that for some
        data-sets some fields are set automatically. Optional fields are
        calculated automatically and the dset is updated - as dset is actually
        an alias (aka pointer), this is reflected at the caller too.
        
        """
        if mode.lower() == 'overwrite':
            # overwrite mode
            try:
                fh = open(self._filename, 'wt')
            except:
                raise Exception('Cannot access the file: ' + self._filename + ' to write to.')
        elif mode.lower() == 'add':
            # add (append) mode
            try:
                fh = open(self._filename, 'at')
            except:
                raise Exception('Cannot access the file: ' + self._filename + ' to write to.')
        else:
            raise Exception('Unknown mode: ' + mode)
        try:
            # Actual writing
            try:
                set_type = dset['type']
            except:
                fh.close()
                raise Exception('Data-set\'s dictionary is missing the required \'type\' key')
            # handle nan or inf
            if 'data' in dset.keys():
                dset['data'] = np.nan_to_num(dset['data'])

            if set_type == 58:
                _write58(fh, dset, mode, _filename=self._filename, force_double=force_double)
            else:
                # Unsupported data-set - do nothing
                pass
        except:
            fh.close()
            raise  # re-raise the last exception
        else:
            fh.close()
        self.refresh()

uff_file=UFF('datasets/Test.uff')

data = uff_file.read_sets()

print(data[4]['x'], data[4]['data'])
if __name__ == '__main__':
    pass