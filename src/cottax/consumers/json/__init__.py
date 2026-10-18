'''
A graph written as JSON, `write_json`, and read back, `read_json` (`format.py`): places by
the `PlaceCodec`, node names by the `LabelCodec` and nodes by the `Spelling` registered for
their class (`interface.py`). Every refusal kind is in `refusals`.
'''

from .format import read_json, write_json

__all__ = ['write_json', 'read_json']
