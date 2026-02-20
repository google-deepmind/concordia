"""Shared test configuration for contrib/components tests."""

import typing


if not hasattr(typing, 'override'):
  typing.override = lambda f: f
