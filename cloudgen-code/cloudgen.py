#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

########################################################################
#    G. Gonçalves, I. Drago, A. B. Vieira, A. P. C. da Silva, J. M.
#    de Almeida, and M. Mellia, “Workload Models and Performance
#    Evaluation of Cloud Storage Services,” RT.DCC.003/2015, UFMG,
#    http://homepages.dcc.ufmg.br/∼ggoncalves/techrpt/15003.pdf.
#
#    Copyright (C) 2015
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
########################################################################


from argparse import ArgumentParser
from collections import defaultdict
from collections import deque
from random import choice
from random import seed as choice_seed
from numpy import random, arange

import collections
import itertools
import heapq
import math
import time

import sys
import os

#******************************************************************************
# Global parameters for the distributions
#******************************************************************************
param = {}

#******************************************************************************
# Usage
#******************************************************************************
def process_opt():
    parser = ArgumentParser()

    parser.add_argument("-i", dest="input", type=str,
              help="Input content sharing network (see the examples for the format)")

    parser.add_argument("-d", dest="num_dv", type=int,
              help="Input number of devices - synthetic network is created")

    parser.add_argument("-n", dest="num_ns", default=None, type=int,
              help="Input number of namespaces - synthetic network is created")

    parser.add_argument("-c", dest="conf", default=None, type=str,
              help="Configuration parameters (see the examples for campuses and PoPs networks)")

    parser.add_argument("-t", dest="timeout", default=30, type=int,
              help="Duration of the simulation in days (default = 30)")

    parser.add_argument("-r", dest="seed", default=int(time.time()), type=int,
              help="Random seed (default unix_time)")

    parser.add_argument("-l", dest="outlab", default="0",
              help="Output content sharing network file label (default = 0)")

    opt = parser.parse_args()

    if not (opt.input or opt.num_dv) or not opt.conf:
        parser.print_help()
        print >> sys.stderr, "\n -c and (-i or -d) are mandatory"
        exit(1)

    execfile(opt.conf, param)
    return opt

#******************************************************************************
# help function
#******************************************************************************
def rpareto(alpha, min_val):
    b = min_val
    a = alpha
    u = random.uniform()
    return (b / ((1-u)**(1/a)))

#******************************************************************************
# Events: priority queue of events (sorted by time)
# https://docs.python.org/2/library/heapq.html
#******************************************************************************
class Events():
    def __init__(self):
        self.pq = []                     # list of entries arranged in a heap
        self.entry_finder = {}           # mapping to entries
        self.REMOVED = '<removed>'       # placeholder for a removed entry
        self.counter = itertools.count() # unique sequence count

    def put(self, event):
        if event in self.entry_finder:
            self.remove_event(event)
        count = next(self.counter)
        entry = [event.time, count, event]
        self.entry_finder[event] = entry
        heapq.heappush(self.pq, entry)

    def remove_event(self, event):
        entry = self.entry_finder.pop(event)
        entry[-1] = self.REMOVED

    def get(self):
        while self.pq:
            time, count, event = heapq.heappop(self.pq)
            if event is not self.REMOVED:
                del self.entry_finder[event]
                return event
        raise KeyError('pop from an empty priority queue')

#******************************************************************************
# Now we define each specific event that can be part of the simulation
#******************************************************************************

#******************************************************************************
# DeviceON Event
#******************************************************************************
class DeviceON(object):
    def next_event(self, base):
        # here we have a complex procedure to determine the inactive time
        ts = time.gmtime(base)
        #idx = (ts.tm_wday * 24) + ts.tm_hour
        idx = ts.tm_hour        

        # one distribution in three ranges
        lnp = param["INTER_SESSION"]

        # probability of each range varying per time of the day/week
        probs = param["INTER_PROB"][idx]
        
        # use short inter-sessions within SHORT_INTER_RANGE period only
        if (ts.tm_hour >= param["SHORT_INTER_RANGE"][0]) and (ts.tm_hour <= param["SHORT_INTER_RANGE"][1]):
            p = random.uniform()
            if p < probs[0]:
                return base + random.lognormal(lnp[0][0], lnp[0][1])
            elif p < probs[0] + probs[1]:
                return base + random.lognormal(lnp[1][0], lnp[1][1])
            else:
                if len(lnp[2]) == 1:
                    return base + rpareto(lnp[2][0], 120000)
                else:
                    return base + random.lognormal(lnp[2][0], lnp[2][1])
        else:
            if self.device.freq:
                return base + random.lognormal(lnp[1][0], lnp[1][1])
            else:
                if len(lnp[2]) == 1:
                    return base + rpareto(lnp[2][0], 120000)
                else:
                    return base + random.lognormal(lnp[2][0], lnp[2][1])                
            

    def __init__(self, t, device):
        self.device = device
        self.time = self.next_event(t)
        self.device.on_time = self.time

    def execute(self):
        self.device.on = True

        # dev_on now - - - host_int - inactive_time
        if self.device.off_time:
            print >> sys.stdout, \
                "dev_on %.3f" % self.time, "- - -", self.device.id, "-", \
                    "%.3f" % (self.time - self.device.off_time)

        else:
            print >> sys.stdout, \
                "dev_on %.3f" % self.time, "- - -", self.device.id, "- -"

        # pick up the random folder
        self.device.active_ns = choice(self.device.namespaces)

        # create the devoff event
        evnts = [DeviceOFF(self.time, self.device)]

        # check whether we create modifications
        if random.uniform() > param["MOD_IN_SESSION"][0]:
            # append the list of modification events
            evnts += \
                ChangeDev.changedev_factory(self.time, self.device, evnts[-1])

        return evnts

#******************************************************************************
# DeviceOFF event
#******************************************************************************
class DeviceOFF(object):
    @staticmethod
    def next_event(base):
        # seconds
        return base + \
            random.lognormal(param["SESSION"][0], param["SESSION"][1])

    def __init__(self, t, device):
        self.device = device
        self.time = self.next_event(t)
        self.device.off_time = self.time

    def execute(self):
        self.device.on = False

        # dev_off now - - - host_int - session_duration
        print >> sys.stdout, \
            "dev_off %.3f" % self.time, "- - -", self.device.id, "-", \
                "%.3f" % (self.time - self.device.on_time)

        # prepare the 'DeviceON' event
        return [DeviceON(self.time, self.device)]

#******************************************************************************
# Generate uploads
#******************************************************************************
class ChangeDev():
    @staticmethod
    def next_event(base):
        # seconds
        return base + \
            random.lognormal(param["INTERMOD"][0], param["INTERMOD"][1])

    # we have a factory method to generate the full list of uploads at once
    @staticmethod
    def changedev_factory(t, device, off_event):

        # matching heuristic
        chg_s = [random.pareto(param["NUMMOD"][0]) * param["NUMMOD"][1]
                 for i in range(1000)]
        dur_s = [random.lognormal(param["SESSION"][0], param["SESSION"][1])
                 for i in range(999)]

        chg_s.sort()
        dur_s.append(off_event.time - t)
        dur_s.sort()
        changes = int(math.ceil(chg_s[dur_s.index(off_event.time - t)]))

        # schedule the uploads, going back to begin if we reach the session end
        evnts = []
        t_next = 0
        for i in range(changes):
            t_next = ChangeDev.next_event(t_next) % (off_event.time - t)
            evnts += [ChangeDev(t + t_next, device)]

        # the list of events
        return evnts

    # next we define the usual events methods
    def __init__(self, t, device):
        self.device = device
        self.time = t

    def execute(self):
        # determine the volume of the change
        if random.uniform() <= param["PROB_VOL"][0]:
            v = int(1000 *
                    random.lognormal(param["VOL"][0][0], param["VOL"][0][1]))
        else:
            v = int(1000 * rpareto(param["VOL"][1][0], 10000))

        #f_changed now - - - host_int - folder vers1 vers2 ses_on vol type
        print >> sys.stdout, \
            "f_changed %.3f" % self.time, "- - -", self.device.id, "-", \
                self.device.active_ns.id, self.device.active_ns.jid, \
                    self.device.active_ns.jid + 1, \
                        "%.3f" % (self.time - self.device.on_time), v, "up -"

        # prepare content propagation events
        evnts = []

        for i in self.device.active_ns.my_devs:
            if i != self.device:
                # if on, propagate content now
                if i.on:
                    evnts += [SyncDevice(self.time, i)]

                # if off, check whether we have already scheduled propagation
                elif not i.updates:
                    evnts += [SyncDevice(i.on_time, i)]

                # put the pending update in the device list
                i.updates.append(Update(self.device.active_ns, v))

        self.device.active_ns.jid += 1

        return evnts

#******************************************************************************
# SyncDevice: perform pending syncs from the content propagation network
#******************************************************************************
class SyncDevice(object):
    def __init__(self, t, device):
        self.device = device
        self.time = t

    def execute(self):
        for ns in set([i.ns for i in self.device.updates]):
            v0 = min([i.jid for i in self.device.updates if i.ns == ns])
            v1 = 1 + max([i.jid for i in self.device.updates if i.ns == ns])
            vol = sum([i.size for i in self.device.updates if i.ns == ns])

            #f_changed now - - - host_int - folder vers1 vers2 ses_on vol type
            print >> sys.stdout, \
                "f_changed %.3f" % self.time, "- - -", \
                    self.device.id, "-", ns.id, v0, v1, \
                      "%.3f" % (self.time - self.device.on_time), vol, "down -"

        self.device.updates = []
        return []

#******************************************************************************
# namespace pending update
#******************************************************************************
class Update(object):
    def __init__(self, ns, size):
        self.ns = ns
        self.jid = ns.jid
        self.size = size

#******************************************************************************
# class representing the namespaces
#******************************************************************************
class NameSpace(object):
    # constructor
    def __init__(self, id, my_devs=None):
        self.id = id

        self.jid = -1
        if my_devs:
            self.my_devs = my_devs
        else:
            self.my_devs = []

    # add a device id to the list of devices of this namespace
    def add_device(self, device):
        self.my_devs.append(device)

#******************************************************************************
# class representing devices
#******************************************************************************
class Device():
    # create a device
    def __init__(self, id):
        # device id
        self.id = id
        
        # frequent device
        self.freq = False

        # pending updates
        self.updates = []

        # keep track of the time devices start a session
        self.on = False
        self.on_time = None
        self.off_time = None

        # hold namespaces
        self.namespaces = []
        self.active_ns = None

    # add a new namespace to this device
    def add_namespace(self, ns):
        self.namespaces.append(ns)
        self.active_ns = ns

    # force the most popular namespace to be the active one
    def select_active(self):
        self.namespaces.sort(key=lambda n: len(n.my_devs))
        self.active_ns = self.namespaces[-1]

#******************************************************************************
# Create the synthetic network
#******************************************************************************
def gen_network(num_dv, num_ns, my_devices, my_namespaces):

    # derive the number of namespace
    dv_s = param["DV_DG"][0]
    dv_p = param["DV_DG"][1]
    nd = 1 + (dv_s * (1.0 - dv_p) / dv_p)

    ns_s = param["NS_DG"][0]
    ns_p = param["NS_DG"][1]
    dn = 1 + (ns_s * (1.0 - ns_p) / ns_p)

    if not num_ns:
        num_ns = int(num_dv * nd / dn)
    print >> sys.stdout, 'network params', dv_s, dv_p, ns_s, ns_p, num_ns

    # sample the number of devices per namespace
    ns_dgr = [x + 1 for x in random.negative_binomial(ns_s, ns_p, num_ns)]

    # sample the number of namespaces per device
    dv_dgr = [x + 1 for x in random.negative_binomial(dv_s, dv_p, num_dv)]

    # create the population of edges leaving namespaces
    l = [i for i, j in enumerate(ns_dgr) for k in range(j)]
    random.shuffle(l)
    ns_pop = deque(l)

    # create empty namespaces
    for ns in range(num_ns):
        my_namespaces[ns] = NameSpace(ns)

    # first we pick a random namespace for each devices
    for dv in range(num_dv):
        ns = ns_pop.pop()
        my_devices[dv] = Device(dv)

        my_devices[dv].add_namespace(my_namespaces[ns])
        my_namespaces[ns].add_device(my_devices[dv])

    # then we complement the namespace degree

    # we skip devices with degree 1 in a first pass, since they just got 1 ns
    r = 1

    # we might have less edges leaving devices than necessary
    while ns_pop:
        # create the population of edges leaving devices
        l = [i for i, j in enumerate(dv_dgr) for k in range(j - r)]
        random.shuffle(l)
        dv_pop = deque(l)

        # if we need to recreate the population, we use devices w/ degree 1 too
        r = 0

        while ns_pop and dv_pop:
            dv = dv_pop.pop()
            ns = ns_pop.pop()

            # we are lazy and skip the unfortunate repetitions
            if not ns in my_devices[dv].namespaces:
                my_devices[dv].add_namespace(my_namespaces[ns])
                my_namespaces[ns].add_device(my_devices[dv])
            else:
                ns_pop.append(ns)

    # initialize the active namespace
    for i in my_devices:
        my_devices[i].select_active()

#******************************************************************************
# implements the simulation
#******************************************************************************
if __name__ == '__main__':
    opt = process_opt()
    print >> sys.stdout, 'Duration (days)', opt.timeout
    print >> sys.stdout, 'Config File', opt.conf

    START = 0 #1388534400 # 01/01/2014

    # simulation start at a day (whatever day)
    t = START
    
    print >> sys.stdout, 'Seed', opt.seed
    random.seed(opt.seed)
    choice_seed(opt.seed)

    # and we start without any events
    events = Events()

    # collection of devices
    my_devices = {}

    # collection of namespaces
    my_namespaces = {}

    # load the network from a trace
    if opt.input:
        with open(opt.input) as topology:
            for line in topology:
                l = map(int, line.split())

                # create a device and its on/off events
                dev = Device(l[0])
                my_devices[l[0]] = dev

                # and load the list of namespaces
                for ns in l[1:]:
                    if ns in my_namespaces:
                        my_namespaces[ns].add_device(dev)
                    else:
                        my_namespaces[ns] = NameSpace(ns, [dev])
                    dev.add_namespace(my_namespaces[ns])

    # or create the synthetic network
    elif opt.num_dv:
        gen_network(opt.num_dv, opt.num_ns, my_devices, my_namespaces)

    # output the network
    print >> sys.stdout, 'Devices', len(my_devices), 'Namespaces', len(my_namespaces)
    f = open('network_sint_' + opt.outlab + '.txt','w')
    for i in my_devices:
        for namespace in my_devices[i].namespaces:
            print >> f, my_devices[i].id, namespace.id
    f.close()

    # define frequent devices at random
    n_freq_devs = int(param["FRAC_FREQ_DEVS"][0] * len(my_devices))
    i_freq_devs = arange( len(my_devices) )
    random.shuffle(i_freq_devs)
    i_freq_devs = set(i_freq_devs[:n_freq_devs])
    i_count = 0
    for i in my_devices:
        if i_count in i_freq_devs:
            my_devices[i].freq = True
        i_count += 1        

    # start up the event simulation by creating dev_on events
    for i in my_devices:
        events.put(DeviceON(t, my_devices[i]))

    # we iterate over events until the simulation time reaches the limit (days)
    while t < START + opt.timeout * 24.0 * 3600.0:
        try:
            event = events.get()
        except KeyError:
            print >> sys.stderr, "No more events to simulate!"
            break
        t = event.time
        for i in event.execute(): events.put(i)
    # and we are done
