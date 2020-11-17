# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"

# have all systems inherit from institute?


class System:
    def __init__(self):
        pass

    def blaa(self, filepath, **kwargs):
        raise NotImplementedError


class Zyla(System):
    raw_pixel_size = 1

    def __init__(self):
        pass


systems_dict = {"zyla": Zyla}


def choose_system(name):
    return systems_dict[name]()
