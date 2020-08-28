# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from server.models.base_model_ import Model
from server import util
import datetime as dt
import threading

class Threshold_ScalingInfo(Model):

    def __init__(self, sfc_name: str=None, scaling_name: str=None, threshold_in: float=None, threshold_out: float=None, interval: float=None, duration: float=None):

        self.swagger_types = {
            'sfc_name': str,
            'scaling_name': str,
            'threshold_in': float,
            'threshold_out': float,
            'interval': float,
            'duration': float
        }

        self.attribute_map = {
            'sfc_name': 'sfc_name',
            'scaling_name': 'scaling_name',
            'threshold_in': 'threshold_in',
            'threshold_out': 'threshold_out',
            'interval': 'interval',
            'duration': 'duration'
        }

        self._sfc_name = sfc_name
        self._scaling_name = scaling_name
        self._threshold_in = threshold_in
        self._threshold_out = threshold_out
        self._interval = interval
        self._duration = duration

    @classmethod
    def from_dict(cls, dikt) -> 'Threshold_ScalingInfo':
        return util.deserialize_model(dikt, cls)

    @property
    def sfc_name(self) -> str:
        return self._sfc_name

    @sfc_name.setter
    def sfc_name(self, sfc_name: str):
        self._sfc_name = sfc_name

    @property
    def scaling_name(self) -> str:
        return self._scaling_name

    @scaling_name.setter
    def scaling_name(self, sfc_name: str):
        self._scaling_name = sfc_name

    @property
    def threshold_in(self) -> float:
        return self._threshold_in

    @threshold_in.setter
    def threshold_in(self, threshold_in: float):
        self._threshold_in = threshold_in

    @property
    def threshold_out(self) -> float:
        return self._threshold_out

    @threshold_out.setter
    def threshold_out(self, threshold_out: float):
        self._threshold_out = threshold_out

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, interval: float):
        self._interval = interval

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float):
        self._duration = duration


class DQN_ScalingInfo(Model):

    def __init__(self, sfc_name: str=None, scaling_name: str=None, slo: float=None, interval: float=None, duration: float=None, has_dataset: bool=None):

        self.swagger_types = {
            'sfc_name': str,
            'scaling_name': str,
            'slo': float,
            'interval': float,
            'duration': float,
            'has_dataset': bool
        }

        self.attribute_map = {
            'sfc_name': 'sfc_name',
            'scaling_name': 'scaling_name',
            'slo': 'slo',
            'interval': 'interval',
            'duration': 'duration',
            'has_dataset': 'has_dataset'
        }

        self._sfc_name = sfc_name
        self._scaling_name = scaling_name
        self._slo = slo
        self._interval = interval
        self._duration = duration
        self._has_dataset = has_dataset

    @classmethod
    def from_dict(cls, dikt) -> 'Threshold_ScalingInfo':
        return util.deserialize_model(dikt, cls)

    @property
    def sfc_name(self) -> str:
        return self._sfc_name

    @sfc_name.setter
    def sfc_name(self, sfc_name: str):
        self._sfc_name = sfc_name

    @property
    def scaling_name(self) -> str:
        return self._scaling_name

    @scaling_name.setter
    def scaling_name(self, sfc_name: str):
        self._scaling_name = sfc_name

    @property
    def slo(self) -> float:
        return self._slo

    @slo.setter
    def slo(self, slo: float):
        self._slo = slo

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, interval: float):
        self._interval = interval

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float):
        self._duration = duration

    @property
    def has_dataset(self) -> bool:
        return self._has_dataset

    @has_dataset.setter
    def has_dataset(self, has_dataset: bool):
        self._has_dataset = has_dataset

class AutoScaler():
    def __init__(self, scaling_info, type):
        self.sfc_name = scaling_info.sfc_name
        self.scaling_name = scaling_info.scaling_name
        self.createdTime = dt.datetime.now()
        self.active_flag = True
        self.type = type
        self.interval = scaling_info.interval
        self.duration = scaling_info.duration

        if type == "threshold":
            self.threshold_in = scaling_info.threshold_in
            self.threshold_out = scaling_info.threshold_out

        elif type == "dqn":
            self.slo = scaling_info.slo
            self.has_dataset = scaling_info.has_dataset

    def get_info(self):
        if self.type == "threshold":
            return {
                "sfc_name": self.sfc_name,
                "scaling_name": self.scaling_name,
                "createdTime": self.createdTime,
                "interval": self.interval,
                "duration": self.duration,
                "active_flag": self.active_flag,
                "type": self.type,
                "threshold_in": self.threshold_in,
                "threshold_out": self.threshold_out
            }
        elif self.type == "dqn":
            return {
                "sfc_name": self.sfc_name,
                "scaling_name": self.scaling_name,
                "createdTime": self.createdTime,
                "interval": self.interval,
                "duration": self.duration,
                "active_flag": self.active_flag,
                "type": self.type,
                "slo": self.slo,
                "has_dataset": self.has_dataset
            }

    def get_sfc_name(self):
        return self.sfc_name

    def set_sfc_name(self, sfc_name):
        self.sfc_name = sfc_name

    def get_scaling_name(self):
        return self.scaling_name

    def set_scaling_name(self, scaling_name):
        self.scaling_name = scaling_name

    def get_createdTime(self):
        return self.createdTime

    def set_createdTime(self, createdTime):
        self.createdTime = createdTime

    def get_createdTime(self):
        return self.createdTime

    def set_createdTime(self, createdTime):
        self.createdTime = createdTime

    def get_active_flag(self):
        return self.active_flag

    def set_active_flag(self, active_flag):
        self.active_flag = active_flag

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval

    def get_duration(self):
        return self.duration

    def set_duration(self, duration):
        self.duration = duration

    def get_threshold_in(self):
        return self.threshold_in

    def set_threshold_in(self, threshold_in):
        self.threshold_in = threshold_in

    def get_threshold_out(self):
        return self.threshold_out

    def set_threshold_out(self, threshold_out):
        self.threshold_out = threshold_out

    def get_slo(self):
        return self.slo

    def set_slo(self, slo):
        self.slo = slo

    def get_has_dataset(self):
        return self.has_dataset

    def set_has_dataset(self, has_dataset):
        self.has_dataset = has_dataset
