'''`Architecting`: the architectures as a surface's methods, closed by the closing it holds.'''

from ...core import CutRule, GaussSeidel, Label
from ...core.frozen import AbstractVar
from ..surface.interface import Part
from .architectures import IDF, MDA, MDF, SAND
from .interface import Closing


class Architecting(Part):
    '''The architectures a study applies, each named after the op it builds, every group of
    cuts closed by `closing`; which variables to cut is the rule, per call.'''

    closing : AbstractVar[Closing]
    '''How a group of cuts is closed: another closing is another surface, or the op written
    in full.'''

    def mdf(self, top: Label, rule: CutRule = GaussSeidel()) -> MDF:
        '''Multidisciplinary feasible under `top`: one MDA per cycle, nested in it.'''
        return MDF(rule, self.closing, top)

    def idf(self, top: Label, rule: CutRule = GaussSeidel()) -> IDF:
        '''Individual discipline feasible under `top`: the closures combined into it.'''
        return IDF(rule, self.closing, top)

    def sand(self, top: Label, rule: CutRule = GaussSeidel()) -> SAND:
        '''Simultaneous analysis and design under `top`: the bodies' states are its own.'''
        return SAND(rule, self.closing, top)

    def mda(self, rule: CutRule = GaussSeidel()) -> MDA:
        '''A multidisciplinary analysis: one closure per cycle, and no design.'''
        return MDA(rule, self.closing)
