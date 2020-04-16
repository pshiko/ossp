from dataclasses import dataclass
from typing import List, Union

import numpy as np
from ortools.sat.python import cp_model


@dataclass
class JobInstance:
    job_id: int
    process_duration: int  # duration of job processing
    limit: int = np.iinfo(np.int32(10)).max  # the time to end the job
    release: int = 0  # the time to be able to start job
    job_name: str = ''


@dataclass
class JobVar:
    job_id: int
    start: cp_model.IntVar
    end: cp_model.IntVar
    delay: cp_model.IntVar
    limit: int
    release: int
    process_durations: List[cp_model.IntervalVar]
    assigned_flags: List[cp_model.IntVar]
    job_name: str = ''


@dataclass
class AssignedJob:
    job_id: int
    start: int
    end: int
    delay: int
    limit: int
    release: int
    assigned_flags: List[Union[int, bool]]
    job_name: str = ''
