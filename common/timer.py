# -*- coding: utf-8 -*-
"""简单分模块计时器（单线程 wall-clock，time.perf_counter）。"""
import time
from collections import OrderedDict


class Timer:
    def __init__(self):
        self.t = OrderedDict()
        self._k = None
        self._t0 = 0.0

    def tic(self, key):
        now = time.perf_counter()
        if self._k is not None:
            self.t[self._k] = self.t.get(self._k, 0.0) + now - self._t0
        self._k, self._t0 = key, now

    def stop(self):
        self.tic(None)
        self._k = None

    def total(self):
        return sum(self.t.values())

    def report(self, title=''):
        tot = self.total()
        lines = [f'--- 计时分解 {title}  总计 {tot:.3f} s ---']
        for k, v in self.t.items():
            lines.append(f'  {k:<22s} {v:9.3f} s  {100 * v / max(tot, 1e-30):6.2f} %')
        return '\n'.join(lines)
