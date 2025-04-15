#! coding: utf-8

from enum import IntEnum


class ScalarType(IntEnum):
    # Integral types
    BOOL = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4

    # floating point types
    FLOAT32 = 5
    FLOAT64 = 6
    FLOAT128 = 7


_dtype_to_str_map = {
    # Integral types
    ScalarType.BOOL: "bool",
    ScalarType.INT8: "int8",
    ScalarType.INT16: "int16",
    ScalarType.INT32: "int",
    ScalarType.INT64: "long",

    # floating point types
    ScalarType.FLOAT32: "float32",
    ScalarType.FLOAT64: "double",
}

_str_to_dtype_map = {v: k for k, v in _dtype_to_str_map.items()}


_dtype_itemsize_map = {
    # Integral types
    ScalarType.BOOL: 1,
    ScalarType.INT8: 1,
    ScalarType.INT16: 2,
    ScalarType.INT32: 4,
    ScalarType.INT64: 8,

    # floating point types
    ScalarType.FLOAT16: 2,
    ScalarType.FLOAT32: 4,
    ScalarType.FLOAT64: 8,
}


def dtype_to_str(dtype: ScalarType) -> str:
    return _dtype_to_str_map[dtype]


def str_to_dtype(dtype_str: str) -> ScalarType:
    return _str_to_dtype_map[dtype_str]


def is_integral_dtype(dtype: ScalarType) -> bool:
    return dtype in (ScalarType.BOOL,
                     ScalarType.INT8,
                     ScalarType.INT16,
                     ScalarType.INT32,
                     ScalarType.INT64)


def is_floating_point_dtype(dtype: ScalarType) -> bool:
    return dtype in (ScalarType.FLOAT16,
                     ScalarType.FLOAT32,
                     ScalarType.FLOAT64)


class DType:
    __slots__ = ("_scalar_type", )

    # Singleton instance
    _instance = {}

    def __new__(cls, dtype: ScalarType):
        if not isinstance(dtype, ScalarType):
            raise ValueError(f"Invalid dtype: {dtype}")

        if dtype not in cls._instance:
            inst = super().__new__(cls)
            inst._scalar_type = dtype
            cls._instance[dtype] = inst

        return cls._instance[dtype]
    
    @property
    def is_integral(self) -> bool:
        return is_integral_dtype(self._scalar_type)
    
    @property
    def is_floating_point(self) -> bool:
        return is_floating_point_dtype(self._scalar_type)
    
    @property
    def name(self) -> str:
        return dtype_to_str(self._scalar_type)

    @property
    def itemsize(self) -> int:
        return _dtype_itemsize_map[self._scalar_type]
    
    def __repr__(self) -> str:
        return f"DType({self.name})"
    
    def __eq__(self, other: object) -> bool:
        return  isinstance(other, DType) and self._scalar_type == other._scalar_type
    
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        return hash(self._scalar_type)


# DType aliases
bool = DType(ScalarType.BOOL)
int8 = DType(ScalarType.INT8)
int16 = DType(ScalarType.INT16)
int32 = DType(ScalarType.INT32)
int64 = DType(ScalarType.INT64)
int = DType(ScalarType.INT32)
long = DType(ScalarType.INT64)

# Floating point types
float16 = DType(ScalarType.FLOAT16)
float32 = DType(ScalarType.FLOAT32)
float64 = DType(ScalarType.FLOAT64)
float = DType(ScalarType.FLOAT32)
double = DType(ScalarType.FLOAT64)