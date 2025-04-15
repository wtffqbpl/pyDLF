#! coding: utf-8

from enum import Enum
from typing import Optional
from ._pydlf import Device as _Device
from ._pydlf import DeviceType as _DeviceType


class DeviceType(Enum):
    """ Device types enumeration. """
    CPU = _DeviceType.CPU
    CUDA = _DeviceType.CUDA


class Device:
    """
    A class representing a compute device (CPU or CUDA).
    This class provides a Pythonic interface to the underlying C++ Device class.
    """

    def __init__(self, device_type: DeviceType, device_index: int = 0):
        """
        Initialize a Device object.

        Args:
            device_type (DeviceType): The type of device to create.
            device_index (int, optional): The index of the device. Defaults to 0.
        """
        self._device = _Device()

        if device_type == DeviceType.CPU:
            self._device = _Device.cpu()
        elif device_type == DeviceType.CUDA:
            self._device = _Device.cuda()
        else:
            raise ValueError(f"Invalid device type: {device_type}")

        self.device_type = device_type
        self.device_index = device_index
    
    @classmethod
    def cpu(cls) -> "Device":
        """
        Create a CPU device.
        """
        return cls(DeviceType.CPU)
    
    @classmethod
    def cuda(cls, device_index: Optional[int] = None) -> "Device":
        """
        Create a CUDA device.
        """
        return cls(DeviceType.CUDA, device_index)
    
    @property
    def index(self) -> int:
        """
        Get the index of the device.
        """
        return self._device.index()
    
    @property
    def is_cpu(self) -> bool:
        """
        Check if the device is a CPU.
        """
        return self._device.is_cpu()
    
    @property
    def is_cuda(self) -> bool:
        """
        Check if the device is a CUDA device.
        """
        return self._device.is_cuda()

    def __str__(self):
        return f"Device(type={self.device_type}, index={self.device_index})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other: "Device") -> bool:
        if not isinstance(other, Device):
            return False
        return self.device_type == other.device_type and self.device_index == other.device_index
    
    def __hash__(self) -> int:
        """
        Get a hash value for the device.
        """
        return hash((self.device_type, self.device_index))
    
    def __ne__(self, other: "Device") -> bool:
        """
        Check if the device is not equal to another device.
        """
        return not self.__eq__(other)
