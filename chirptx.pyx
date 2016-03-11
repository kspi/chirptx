import pandas
import numpy
cimport numpy


# Durations are specified in seconds
DEF TIME_STEP = 0.01
DEF TX_DURATION = 0.04

cdef class Chirp:
    cdef bint is_transmitting
    cdef double transmission_began_at
    cdef bint transmission_failed

    def __init__(self):
        self.is_transmitting = False
        self.transmission_began_at = 0
        self.transmission_failed = False

    cdef bint should_begin_transmission(self, double t):
        raise RuntimeError("Abstract method called.")

    cdef bint update(self, double t):
        """Returns true if transmission began."""
        if self.is_transmitting:
            if t - self.transmission_began_at > TX_DURATION:
                self.is_transmitting = False
        else:
            if self.should_begin_transmission(t):
                self.is_transmitting = True
                self.transmission_began_at = t
                self.transmission_failed = False
                return True
        return False

    cdef bint fail(self):
        """Returns true if transmission failed."""
        cdef bint already_failed = self.transmission_failed
        self.transmission_failed = True
        return not already_failed

cdef class Periodic(Chirp):
    cdef double period
    cdef double last_transmission

    def __init__(self, double period):
        super().__init__()
        self.period = period
        self.last_transmission = -int(numpy.random.rand() * period)

    cdef bint should_begin_transmission(self, double t):
        if self.last_transmission + self.period < t:
            self.last_transmission = t
            return True
        else:
            return False

cdef class RandomizedPeriodic(Chirp):
    cdef double period
    cdef double spread
    cdef double next_transmission

    def __init__(self, double period, double spread):
        super().__init__()
        self.period = period
        self.spread = spread
        self.schedule(0)

    cdef void schedule(self, double t):
        cdef double this_period = self.period - self.spread / 2 + numpy.random.rand() * self.spread
        self.next_transmission = t + this_period

    cdef bint should_begin_transmission(self, double t):
        if self.next_transmission <= t:
            self.schedule(t)
            return True
        else:
            return False

class Result:
    pass


cpdef simulate(numpy.ndarray actors, numpy.ndarray[double] delays, double duration):
    cdef numpy.ndarray[double] ts = \
        numpy.arange(0, duration, TIME_STEP, dtype=numpy.float)
    #cdef numpy.ndarray[long] history = \
    #    numpy.zeros(len(ts), dtype=numpy.int)
    cdef int transmissions = 0
    cdef int failures = 0
    cdef int count
    cdef int ti, actor_i
    cdef double t
    cdef double actor_delay
    cdef Chirp actor
    for ti, t in enumerate(ts):
        count = 0
        for actor_i, actor in enumerate(actors):
            actor_delay = delays[actor_i]
            if t >= actor_delay:
                actor_t = t - actor_delay
                transmissions += actor.update(actor_t)
                if actor.is_transmitting:
                    count += 1
        if count > 1:
            for actor in actors:
                failures += actor.fail()
        #history[ti] = count

    result = Result()
    result.transmissions = transmissions
    result.failures = failures
    #result.history = pandas.Series(history, index=pandas.to_datetime(ts, unit='s'))
    return result

