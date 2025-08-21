"""
Package containing task implementations for various robotic environments.

This package provides custom task implementations for reinforcement learning
with Isaac Lab, including specialized configurations for different robots
and training scenarios.
"""

import os
import toml

from isaaclab_tasks.utils import import_packages

from .go2 import *