import re

PYTHON_KEYWORDS = [
    # default python keywords
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 
    'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 
    'with', 'yield', 
    # built-in python functions
    'abs', 'delattr', 'hash', 'memoryview', 'set', 'all', 'dict', 'help', 'min', 
    'setattr', 'any', 'dir', 'hex', 'next', 'slice', 'ascii', 'divmod', 'id', 
    'object', 'sorted', 'bin', 'enumerate', 'input', 'oct', 'staticmethod', 'bool', 
    'eval', 'int', 'open', 'str', 'breakpoint', 'exec', 'isinstance', 'ord', 'sum', 
    'bytearray', 'filter', 'issubclass', 'pow', 'super', 'bytes', 'float', 'iter', 
    'print', 'tuple', 'callable', 'format', 'len', 'property', 'type', 'chr', 
    'frozenset', 'list', 'range', 'vars', 'classmethod', 'getattr', 'locals', 'repr', 
    'zip', 'compile', 'globals', 'map', 'reversed', '__import__', 'complex', 'hasattr', 
    'max', 'round', '__eq__', '__lt__', '__le__', '__gt__', '__ge__', '__add__', '__rsub__', 
    '__imul__', '__iand__', '__debug__', '__bool__', '__len__', '__contains__', '__index__',
    # error types
    'NotImplemented', 'NotImplementedError', 'TypeError', 'ValueError',
    # list functions
    'index', 'count', 'append', 'clear', 'copy', 'extend', 'insert', 'pop', 'remove', 
    'reverse', 'sort', 
    # str functions
    'capitalize', 'casefold', 'center', 'encode', 'endswith', 'expandtabs', 'find', 
    'format', 'format_map', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit',
    'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 
    'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 
    'removeprefix', 'removesuffix', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition'
    'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title',
    'translate', 'upper', 'zfill',
    # hex functions
    'fromhex', 
    # bytes functions,
    'decode', 
    # memoryview functions,
    'tobytes', 'tolist', 'toreadonly', 'release', 'cast', 'nbytes', 'readonly', 'itemsize',
    'ndim', 'shape', 'strides', 'suboffsets', 'c_contiguous', 'contiguous',
    # set functions,
    'isdisjoint', 'issubset', 'union', 'intersection', 'difference', 'symmetric_difference', 
    'copy', 'update', 'intersection_update', 'difference_update', 'symmetric_difference_update',
    'add', 'remove', 'discard', 'clear',
    # dict functions,
    'fromkeys', 'get', 'items', 'keys', 'popitem', 'setdefault', 'values', '__getitem__', '__dict__',
    # contextmanager functions
    '__enter__', '__exit__', 
    # special
    '__args__', '__parameters__', '__class__', '__bases__', '__name__', '__subclasses__', 
    '__main__',
]   


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
