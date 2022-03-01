# objects in __main__ namespace cannot be serialized
from examples.Experiments.FooExp_SubprocessWriterTest import FooExperiment
from labplatform.core.Subject import load_cohort, Subject
import logging

log = logging.getLogger()
log.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)

# load subject cohort
sl = load_cohort('example')
'''
# create a new subject and add it into the cohort
new_sub = Subject(name='T2', group='test', species='Mouse')
sl.add_subject(new_sub)
'''

fe = FooExperiment(subject=sl.subjects[0])
# fe = FooExperiment(subject=new_sub)
fd = fe.devices['FooDevice']
