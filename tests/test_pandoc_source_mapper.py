#!/usr/bin/python3

import os.path
import sys
import unittest
import warnings

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from uberwriter.pandoc_source_mapper import PandocSourceMapper
from uberwriter.settings import Settings


class TestPandocSourceMapper(unittest.TestCase):

    def test_warnings(self):
        directory = os.path.join(os.path.dirname(__file__), "resources", "pandoc_source_mapper")
        for path in [os.path.join(directory, f) for f in os.listdir(directory)]:
            with open(path, "r") as file:
                print("Testing {} in {}".format(
                    repr(os.path.basename(path)),
                    repr(Settings.new().get_value('input-format').get_string())))
                with warnings.catch_warnings(record=True) as w:
                    PandocSourceMapper(file.read()).walk()
                    assert len(w) == 0


if __name__ == '__main__':    
    unittest.main()
