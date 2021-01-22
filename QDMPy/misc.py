# -*- coding: utf-8 -*-

"""
Collection of helper functions that don't belong anywhere else.

Currently functions for loading json files to dicts and the inverse.

Functions
---------
 - `QDMPy.misc.failfloat`
 - `QDMPy.misc.defaultdict_from_d`
 - `QDMPy.misc.json_to_dict`
 - `QDMPy.misc.dict_to_json`
 - `QDMPy.misc.prettyjson`
 - `QDMPy.misc.getsubitems`
 - `QDMPy.misc.basictype2str`
 - `QDMPy.misc.indentitems`
 - `QDMPy.misc.json_remove_comments`
"""

# ============================================================================


__author__ = "Sam Scholten"

# ============================================================================

import os
import warnings
from collections import OrderedDict, defaultdict
import re
import simplejson as json
import numpy as np

# ============================================================================


def failfloat(a):
    """Used in particular for reading the metadata to convert all numbers into
    floats and leave strings as strings.
    """
    try:
        return float(a)
    except ValueError:
        return a


# ============================================================================


def defaultdict_from_d(d):
    "converts d to a defaultdict, with default value of None for all keys"
    dd = defaultdict(lambda: None)
    dd.update(d)
    return dd


# ============================================================================


def json_to_dict(filepath, hook="od"):
    """ read the json file at filepath into a dict """
    _, pattern = os.path.splitext(filepath)
    if pattern != ".json":
        warnings.warn("input options file did not have a json pattern/suffix")

    with open(filepath, "r", encoding="utf-8") as fp:
        if hook == "od":
            oph = OrderedDict
        elif hook == "dd":
            oph = defaultdict_from_d
        else:
            raise RuntimeError("bad choice for dict hook")

        jstring = json_remove_comments(fp.read())
        dct = json.loads(jstring, object_pairs_hook=oph)
        return dct.copy()


# ============================================================================


def dict_to_json(dictionary, filename, path_to_dir=None):
    """ save the dict as a json in a pretty way """
    # ensure json pattern
    root, pattern = os.path.splitext(filename)
    if pattern != ".json":
        warnings.warn("reformatted param save filename to '.json' pattern")
        pattern = ".json"
    filename = root + pattern
    # get cwd
    cwd = os.getcwd()
    if path_to_dir is None:
        filepath = os.path.join(cwd, filename)
    elif not os.path.isdir(path_to_dir):
        warnings.warn(
            "path_to_dir was not a valid directory, instead "
            + "saving to current working directory"
        )
        filepath = os.path.join(cwd, filename)
    else:
        filepath = os.path.join(path_to_dir, filename)

    with open(filepath, "w") as fp:
        fp.write(prettyjson(dictionary))


# ============================================================================


def prettyjson(obj, indent=4, maxlinelength=80):
    """Renders JSON content with indentation and line splits/concatenations to
    fit maxlinelength. Only dicts, lists and basic types are supported.
    <Pass the dict as obj and get back a string>"""

    items, _ = getsubitems(obj, itemkey="", islast=True, maxlinelength=maxlinelength)
    res = indentitems(items, indent, indentcurrent=0)
    return res


# ============================================================================


def getsubitems(obj, itemkey, islast, maxlinelength):
    items = []
    # assume we can concatenate inner content unless a child node returns an
    # expanded list
    can_concat = True

    if isinstance(obj, np.array):
        obj = obj.tolist()

    isdict = isinstance(obj, dict)
    islist = isinstance(obj, list)
    istuple = isinstance(obj, tuple)

    # building json content as a list of strings or child lists
    if isdict or islist or istuple:
        if isdict:
            opening, closing, keys = ("{", "}", iter(obj.keys()))
        elif islist:
            opening, closing, keys = ("[", "]", range(0, len(obj)))
        elif istuple:
            # tuples are converted into json arrays
            opening, closing, keys = ("[", "]", range(0, len(obj)))

        if itemkey != "":
            opening = itemkey + ": " + opening
        if not islast:
            closing += ","

        # Get list of inner tokens as list
        count = 0
        subitems = []
        itemkey = ""
        for k in keys:
            count += 1
            islast_ = count == len(obj)
            itemkey_ = ""
            if isdict:
                itemkey_ = basictype2str(k)
            # inner = (items, indent)
            inner, can_concat_ = getsubitems(obj[k], itemkey_, islast_, maxlinelength)
            # inner can be a string or a list
            subitems.extend(inner)
            # if a child couldn't concat, then we are not able either
            can_concat = can_concat and can_concat_

        # atttempt to concat subitems if all fit within maxlinelength
        if can_concat:
            totallength = 0
            for item in subitems:
                totallength += len(item)
            totallength += len(subitems) - 1  # spaces between items
            if totallength <= maxlinelength:
                str = ""
                # add space between items, comma is already there
                for item in subitems:
                    str += item + " "
                str = str.strip()
                # wrap concatenated content in a new list
                subitems = [str]
            else:
                can_concat = False

        # attempt to concat outer brackets + inner items
        if can_concat:
            if len(opening) + totallength + len(closing) <= maxlinelength:
                items.append(opening + subitems[0] + closing)
            else:
                can_concat = False

        if not can_concat:
            items.append(opening)  # opening brackets
            # Append children to parent list as a nested list
            items.append(subitems)
            items.append(closing)  # closing brackets

    else:
        # basic types
        strobj = itemkey
        if strobj != "":
            strobj += ": "
        strobj += basictype2str(obj)
        if not islast:
            strobj += ","
        items.append(strobj)

    return items, can_concat


# ============================================================================


def basictype2str(obj):
    """This is a filter on objects that get sent to the json. Some types
    can't be stored literally in json files, so we can adjust for that here.
    """
    if isinstance(obj, str):
        strobj = '"' + str(obj) + '"'
    elif isinstance(obj, type(None)):
        strobj = "null"
    elif isinstance(obj, bool):
        strobj = {True: "true", False: "false"}[obj]
    else:
        strobj = str(obj)
    return strobj


# ============================================================================


def indentitems(items, indent, indentcurrent):
    """Recursively traverses the list of json lines, adds indentation based
    on the current depth"""
    res = ""
    indentstr = " " * indentcurrent
    for item in items:
        if isinstance(item, list):
            res += indentitems(item, indent, indentcurrent + indent)
        else:
            res += indentstr + item + "\n"
    return res


# ============================================================================


def json_remove_comments(string, strip_space=True):
    tokenizer = re.compile('"|(/\*)|(\*/)|(//)|\n|\r')
    end_slashes_re = re.compile(r"(\\)*$")

    in_string = False
    in_multi = False
    in_single = False

    new_str = []
    index = 0

    for match in re.finditer(tokenizer, string):

        if not (in_multi or in_single):
            tmp = string[index : match.start()]
            if not in_string and strip_space:
                # replace white space as defined in standard
                tmp = re.sub("[ \t\n\r]+", "", tmp)
            new_str.append(tmp)
        elif not strip_space:
            # Replace comments with white space so that the JSON parser reports
            # the correct column numbers on parsing errors.
            new_str.append(" " * (match.start() - index))

        index = match.end()
        val = match.group()

        if val == '"' and not (in_multi or in_single):
            escaped = end_slashes_re.search(string, 0, match.start())

            # start of string or unescaped quote character to end string
            if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):  # noqa
                in_string = not in_string
            index -= 1  # include " character in next catch
        elif not (in_string or in_multi or in_single):
            if val == "/*":
                in_multi = True
            elif val == "//":
                in_single = True
        elif val == "*/" and in_multi and not (in_string or in_single):
            in_multi = False
            if not strip_space:
                new_str.append(" " * len(val))
        elif val in "\r\n" and not (in_multi or in_string) and in_single:
            in_single = False
        elif not ((in_multi or in_single) or (val in " \r\n\t" and strip_space)):  # noqa
            new_str.append(val)

        if not strip_space:
            if val in "\r\n":
                new_str.append(val)
            elif in_multi or in_single:
                new_str.append(" " * len(val))

    new_str.append(string[index:])
    return "".join(new_str)
