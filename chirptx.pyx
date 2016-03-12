# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
import pandas
import numpy
cimport numpy


cdef class Chirp:
    cdef double transmission_duration
    cdef bint is_transmitting
    cdef double transmission_began_at
    cdef bint transmission_failed

    def __init__(self, double transmission_duration):
        self.transmission_duration = transmission_duration
        self.is_transmitting = False
        self.transmission_began_at = 0
        self.transmission_failed = False

    cdef bint should_begin_transmission(self, double t):
        raise RuntimeError("Abstract method called.")

    cdef bint update(self, double t):
        """Returns true if transmission began."""
        if self.is_transmitting:
            if t - self.transmission_began_at > self.transmission_duration:
                self.is_transmitting = False
        else:
            if self.should_begin_transmission(t):
                self.is_transmitting = True
                self.transmission_began_at = t
                self.transmission_failed = False
                return True
        return False

    cdef bint fail(self):
        """Returns true if transmission has failed."""
        if self.is_transmitting and not self.transmission_failed:
            self.transmission_failed = True
            return True
        else:
            return False


cdef class Periodic(Chirp):
    cdef double period
    cdef double next_transmission

    def __init__(self, double transmission_duration, double period):
        super().__init__(transmission_duration)
        self.period = period
        self.schedule(0)

    cdef double sleep_time(self):
        return self.period

    cdef void schedule(self, double t):
        self.next_transmission = t + self.sleep_time()

    cdef bint should_begin_transmission(self, double t):
        if self.next_transmission <= t:
            self.schedule(t)
            return True
        else:
            return False


cdef class UniformSpread(Periodic):
    cdef double spread

    def __init__(self, double transmission_duration, double mean_period, double spread):
        self.spread = spread
        super().__init__(transmission_duration, mean_period)

    cdef double sleep_time(self):
        return self.period - self.spread / 2 + numpy.random.rand() * self.spread

cdef class Poisson(Periodic):
    cdef double sleep_time(self):
        return numpy.random.exponential(self.period)


class Result:
    pass


cpdef simulate(double duration, double time_step, numpy.ndarray actors, numpy.ndarray[double] delays):
    cdef numpy.ndarray[double] ts = \
        numpy.arange(0, duration, time_step, dtype=numpy.float)
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

