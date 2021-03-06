# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
### BEGIN LICENSE
# Copyright (C) 2012, Wolf Vollprecht <w.vollprecht@gmail.com>
# This program is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License version 3, as published 
# by the Free Software Foundation.
# 
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranties of 
# MERCHANTABILITY, SATISFACTORY QUALITY, or FITNESS FOR A PARTICULAR 
# PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along 
# with this program.  If not, see <http://www.gnu.org/licenses/>.
### END LICENSE
import sys

import locale
import os

import gettext
from gettext import gettext as _

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk # pylint: disable=E0611

from . import UberwriterWindow
from uberwriter_lib import AppWindow

def main():
    'constructor for your class instances'
    # (options, args) = parse_options()
    
    # Run the application.
    app = AppWindow.Application()
    
    # ~ if args:
        # ~ for arg in args:
            # ~ pass 
    # ~ else:
        # ~ pass
    # ~ if options.experimental_features:
        # ~ window.use_experimental_features(True)
        
    app.run(sys.argv)
    
