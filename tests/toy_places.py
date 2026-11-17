'''
The prefixes of a toy place: `Place` asks a place to say its own, and a toy built from its
`segments` says them by building each prefix on every ask -- what a real place keeps instead.
'''


class Segmented:
    '''A place's `prefixes`, from its `segments`: the root, `.x`, `.x.y` of `.x.y.z`.'''

    @property
    def prefixes(self) -> tuple:
        cls, segments = type(self), self.segments
        return tuple(cls(segments[:i]) for i in range(len(segments)))
