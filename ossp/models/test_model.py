from ..data.job import JobInstance
from .model import OSSP

test_jobs = [
    JobInstance(job_id=0, process_duration=466, limit=76, job_name='A'),
    JobInstance(job_id=1, process_duration=53, limit=158, job_name='B'),
    JobInstance(job_id=2, process_duration=525, limit=353, job_name='C'),
    JobInstance(job_id=3, process_duration=580, limit=322, job_name='D'),
    JobInstance(job_id=4, process_duration=2238, limit=968, job_name='E'),
    JobInstance(job_id=5, process_duration=1555, limit=761, job_name='F'),
    JobInstance(job_id=6, process_duration=2038, limit=1183, job_name='G'),
]


def test_minimize_delayed_time():
    model = OSSP(machine_nums=3)
    [model.add_job_instance(job) for job in test_jobs]
    status, results, obj_var = model.minimize_delayed_time()
    print(f'status: {status}')
    print(f'results: {results}')
    print(f'objective: {obj_var}')
    assert obj_var == sum([job.delay for job in results])


def test_minimize_maximum_delayed_time():
    model = OSSP(machine_nums=3)
    [model.add_job_instance(job) for job in test_jobs]
    status, results, obj_var = model.minimize_maximum_delayed_time()
    print(f'status: {status}')
    print(f'results: {results}')
    print(f'objective: {obj_var}')
    assert obj_var == max([job.delay for job in results])
