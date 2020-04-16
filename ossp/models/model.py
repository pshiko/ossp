from logging import getLogger, StreamHandler, Formatter, INFO
from typing import List, Optional

from ortools.sat.python import cp_model

from ossp.data.job import JobInstance, JobVar, AssignedJob

logger = getLogger(__name__)
sh = StreamHandler()
sh.setFormatter(
    Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
)
sh.setLevel(INFO)
logger.addHandler(sh)


class OSSPSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, job_vars: List[JobVar]):
        super().__init__()
        self.solution_count = 0
        self.solutions: List[List[AssignedJob]] = []
        self.job_vars = job_vars

    def on_solution_callback(self):
        self.solution_count += 1
        assigned_jobs = []
        for job_var in self.job_vars:
            assigned_jobs.append(
                AssignedJob(
                    job_id=job_var.job_id,
                    start=self.Value(job_var.start),
                    end=self.Value(job_var.end),
                    delay=self.Value(job_var.delay),
                    limit=self.Value(job_var.limit),
                    release=job_var.release,
                    assigned_flags=[self.Value(flag) for flag in job_var.assigned_flags],
                    job_name=job_var.job_name,
                )
            )
        self.solutions.append(assigned_jobs)

    def solution_count(self):
        return self.solution_count


class OSSP:
    def __init__(self, machine_nums: int = 2):
        self.model: Optional[cp_model.CpModel] = None
        self.job_instances: List[JobInstance] = []
        self.job_vars: List[JobVar] = []
        self.horizon: Optional[int] = None
        self.machine_nums: int = machine_nums
        self.solver = cp_model.CpSolver()
        self.solution_callbacks: Optional[OSSPSolutionPrinter] = None

    def add_job_instance(self, job: JobInstance):
        self.job_instances.append(job)

    def _add_job_var(self, job: JobInstance):
        start = self.model.NewIntVar(
            lb=0, ub=self.horizon,
            name=f'job_start_{job.job_id}',
        )
        end = self.model.NewIntVar(
            lb=0, ub=self.horizon,
            name=f'job_end_{job.job_id}',
        )
        delay = self.model.NewIntVar(
            lb=0, ub=self.horizon,
            name=f'job_delay_{job.job_id}',
        )
        assigned_flags = []
        process_durations = []
        for machine_id in range(self.machine_nums):
            assigned_flag = self.model.NewIntVar(
                0, 1,
                f'machine_{machine_id}_job_assigned_flags_{job.job_id}',
            )
            process_duration = self.model.NewOptionalIntervalVar(
                start=start, size=job.process_duration,
                end=end, is_present=assigned_flag,
                name=f'machine_{machine_id}_job_process_duration_{job.job_id}',
            )
            assigned_flags.append(assigned_flag)
            process_durations.append(process_duration)
        self.job_vars.append(JobVar(
            job_id=job.job_id,
            start=start,
            end=end,
            delay=delay,
            limit=job.limit,
            release=job.release,
            process_durations=process_durations,
            assigned_flags=assigned_flags,
            job_name=job.job_name,
        ))

    def build_constrain(self):
        # job processed after release
        for job in self.job_vars:
            self.model.Add(
                job.release <= job.start
            )

        # constrain for delay time
        for job in self.job_vars:
            self.model.Add(
                (job.end - job.limit) <= job.delay
            )

        # one job is processed only by one machine
        for job in self.job_vars:
            self.model.Add(
                sum(job.assigned_flags) == 1
            )

        # one machine process only one job at a time
        for machine_id in range(self.machine_nums):
            self.model.AddNoOverlap(
                [job.process_durations[machine_id] for job in self.job_vars]
            )

    def _initialize_model(self):
        self.model = cp_model.CpModel()
        self.job_vars = []
        self.horizon = sum([job.process_duration for job in self.job_instances])
        for job in self.job_instances:
            self._add_job_var(job)
        self.build_constrain()
        self.solution_callbacks = OSSPSolutionPrinter(self.job_vars)

    def minimize_delayed_time(self) -> (str, List[AssignedJob], int):
        self._initialize_model()
        self.model.Minimize(
            sum([job.delay for job in self.job_vars])
        )
        status = self.solver.SolveWithSolutionCallback(self.model, self.solution_callbacks)
        result = self.solution_callbacks.solutions[0]
        objective_value = self.solver.ObjectiveValue()
        return self.solver.StatusName(status), result, objective_value

    def minimize_maximum_delayed_time(self) -> (str, List[AssignedJob], int):
        self._initialize_model()
        objective_var = self.model.NewIntVar(lb=0, ub=self.horizon, name='objective')
        self.model.AddMaxEquality(
            objective_var,
            [job.delay for job in self.job_vars]
        )
        self.model.Minimize(objective_var)

        status = self.solver.SolveWithSolutionCallback(self.model, self.solution_callbacks)
        result = self.solution_callbacks.solutions[0]
        objective_value = self.solver.ObjectiveValue()
        return self.solver.StatusName(status), result, objective_value
