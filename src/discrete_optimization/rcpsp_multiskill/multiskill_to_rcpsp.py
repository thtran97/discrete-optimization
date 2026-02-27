#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import (
    MultiskillRcpspProblem,
    PrecomputeEmployeesForTasks,
)

logger = logging.getLogger(__name__)


class MultiSkillToRcpsp:
    def __init__(self, multiskill_model: MultiskillRcpspProblem):
        self.multiskill_model = multiskill_model
        self.worker_type_to_worker = None

    def is_compatible(
        self,
        task_requirements: dict[str, int],
        ressource_availability: dict[str, np.array],
        duration_task,
        horizon,
    ):
        non_zeros_res = [r for r in task_requirements if task_requirements[r] >= 0]
        p = np.multiply.reduce(
            [
                ressource_availability[non_zeros_res[j]][:horizon]
                >= task_requirements[non_zeros_res[j]]
                for j in range(len(non_zeros_res))
            ]
        )
        s = np.sum(p)
        return s >= duration_task

    def _enumerate_task(
        self,
        task,
        params_cp: ParametersCp,
        one_worker_type_per_task: bool,
        check_resource_compliance: bool,
        map_names_to_understandable: dict,
        worker_type_name: list,
        resources_dict: dict,
        initial_mode_details: dict,
    ) -> tuple[str, list[dict], list]:
        """
        Enumerate and filter worker-type assignment modes for a single task.

        Returns:
        - task (str): The task identifier
        - tt (list[dict]): List of filtered task requirement dicts (one per mode)
        - pruned_results (list): List of raw CP enumeration results that passed overskill filtering
        """
        task_solver = PrecomputeEmployeesForTasks(
            ms_rcpsp_problem=self.multiskill_model, cp_solver_name=CpSolverName.CHUFFED
        )

        # Enumerate all valid worker-type assignments via CP
        task_solver.init_model(
            tasks_of_interest=[task],
            consider_units=False,
            consider_worker_type=True,
            one_ressource_per_task=self.multiskill_model.one_unit_per_task_max,
            one_worker_type_per_task=one_worker_type_per_task,
        )
        results = task_solver.solve(
            parameters_cp=params_cp,
            time_limit=5,
            all_solutions=False,
            nr_solutions=100,
        )
        logger.debug(f"  Task {task}: Enumeration found {len(results)} solutions")

        # Keep only minimum-overskill solutions
        if results:
            best_overskill = min(results, key=lambda x: x.overskill_type).overskill_type
            pruned_results = [r for r in results if r.overskill_type == best_overskill]
            logger.debug(
                f"  Task {task}: Filtered to {len(pruned_results)} solutions "
                f"(min overskill={best_overskill})"
            )
        else:
            pruned_results = []
            logger.warning(f"  Task {task}: No feasible solutions found!")

        # Build task_requirement_list from pruned results
        task_requirement_list = []
        for i in range(len(pruned_results)):
            ddd = initial_mode_details[task][pruned_results[i].mode_dict[task]]
            ddd = {
                key: ddd[key]
                for key in ddd
                if key not in self.multiskill_model.skills_set
            }
            wtype_used = [
                (j, pruned_results[i].worker_type_used[j][0])
                for j in range(len(pruned_results[i].worker_type_used))
            ]
            index_non_zeros = [k for k in wtype_used if k[1] > 0]

            task_requirement = ddd
            for k, c in index_non_zeros:
                task_requirement[map_names_to_understandable[worker_type_name[k]]] = c
            task_requirement_list.append(task_requirement)

        # Filter by resource compliance (read-only access to resources_dict)
        if check_resource_compliance:
            tt = [
                t
                for t in task_requirement_list
                if self.is_compatible(
                    task_requirements={r: t[r] for r in t if r != "duration"},
                    duration_task=t["duration"],
                    horizon=self.multiskill_model.horizon,
                    ressource_availability=resources_dict,
                )
            ]
            logger.debug(
                f"  Task {task}: Compliance check: {len(tt)}/{len(task_requirement_list)} modes remain"
            )
        else:
            tt = task_requirement_list

        return task, tt, pruned_results

    def construct_rcpsp_by_worker_type(
        self,
        limit_number_of_mode_per_task: bool = True,
        max_number_of_mode: int = None,
        check_resource_compliance: bool = True,
        one_worker_type_per_task: bool = False,
        nb_workers: int = None,
    ):
        """
        Construct an RCPSP problem by abstracting employees as worker types.
        Precisely, this performs a 3-stage decomposition:
        - Stage 1: Cluster employees into worker types by skill profile
        - Stage 2: Enumerate and select feasible worker-type assignments per task
        - Stage 3: Build the final RCPSP problem based on selected modes and worker-type availability
        """
        start = time.time()
        logger.info("#1 : Cluster employees into worker types by skill profile")
        
        params_cp = ParametersCp.default()
        params_cp.intermediate_solution = True
        
        # Initialize solver
        logger.debug(f"{time.time() - start:.2f}s: Initializing PrecomputeEmployeesForTasks solver...")
        solver = PrecomputeEmployeesForTasks(
            ms_rcpsp_problem=self.multiskill_model, cp_solver_name=CpSolverName.CHUFFED
        )
        logger.debug(f"{time.time() - start:.2f}s: Solver initialized. Found {len(solver.skills_dict)} worker types")
        
        # Stage 1: Clustering
        worker_type_name = sorted(solver.skills_dict)
        worker_type_container = solver.skills_representation_str
        self.skills_dict = solver.skills_dict
        self.skills_representation_str = solver.skills_representation_str
        
        logger.debug(f"{time.time() - start:.2f}s: Clustering {len(self.multiskill_model.employees)} employees into {len(worker_type_name)} worker types")
        calendar_worker_type = {}
        map_names_to_understandable = {
            worker_type_name[i]: "WorkerType-" + str(i)
            for i in range(len(worker_type_name))
        }
        self.map_names_to_understandable = map_names_to_understandable
        self.worker_type_to_worker = {
            self.map_names_to_understandable[k]: solver.skills_representation_str[k]
            for k in self.map_names_to_understandable
        }
        
        # Log worker type details
        if logger.isEnabledFor(logging.DEBUG):
            for worker_type in worker_type_name:
                employees = list(worker_type_container[worker_type])
                readable_name = map_names_to_understandable[worker_type]
                skill_profile = worker_type
                logger.debug(f"{time.time() - start:.2f}s:   {readable_name}: {len(employees)} employee(s) with skills {skill_profile}")
        
        # Calculate aggregate availability
        logger.debug(f"{time.time() - start:.2f}s: Computing aggregate availability calendars for each worker type...")
        for worker_type in worker_type_name:
            employees = list(worker_type_container[worker_type])
            calend = np.array(
                self.multiskill_model.employees[employees[0]].calendar_employee,
                dtype=np.int_,
            )
            for j in range(1, len(employees)):
                calend += np.array(
                    self.multiskill_model.employees[employees[j]].calendar_employee,
                    dtype=np.int_,
                )
            calendar_worker_type[map_names_to_understandable[worker_type]] = calend
                
        # Prepare resources dictionary
        resources_dict = self.multiskill_model.resources_availability
        usage_worker_in_chosen_modes = {k: 0 for k in calendar_worker_type}
        for k in calendar_worker_type:
            resources_dict[k] = calendar_worker_type[k]

        stage1_time = time.time() - start
        
        # Stage 2: Mode enumeration and selection
        effective_workers = nb_workers if nb_workers is not None else os.cpu_count()
        logger.info(f"{time.time() - start:.2f}s: Processing {len(self.multiskill_model.tasks_list)} tasks "
                    f"(parallel threads: {effective_workers})...")
        logger.info(f"{time.time() - start:.2f}s:   Limiting modes per task: {limit_number_of_mode_per_task}")
        if limit_number_of_mode_per_task:
            logger.info(f"{time.time() - start:.2f}s:   Max modes per task: {max_number_of_mode}")
        logger.info(f"{time.time() - start:.2f}s:   Check resource compliance: {check_resource_compliance}")

        initial_mode_details = self.multiskill_model.mode_details
        mode_details_post_compute = {}
        dictionnary_precompute = {}
        task_times = {}

        # ------------------------------------------------------------------
        # Parallel processing: enumerate, filter, compliance
        # Each task gets its own PrecomputeEmployeesForTasks instance so there
        # is no shared mutable state between threads.
        # ------------------------------------------------------------------
        tasks_list = self.multiskill_model.tasks_list
        # Dict to accumulate (task_requirement_list_filtered, pruned_results) per task
        enumeration_results: dict = {}

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._enumerate_task,
                    task,
                    params_cp,
                    one_worker_type_per_task,
                    check_resource_compliance,
                    map_names_to_understandable,
                    worker_type_name,
                    resources_dict,
                    initial_mode_details,
                ): task
                for task in tasks_list
            }

            progress = tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Enumerating modes per task",
                unit="task",
                dynamic_ncols=True,
            )
            for future in progress:
                t_start = time.time()
                task = future_to_task[future]
                result_task, tt_filtered, pruned = future.result()
                enumeration_results[result_task] = (tt_filtered, pruned)
                task_times[task] = time.time() - t_start

        # ------------------------------------------------------------------
        # Load-balancing and truncation
        # ------------------------------------------------------------------
        logger.info(f"{time.time() - start:.2f}s: Applying load-balancing heuristic and truncation to modes...")
        for task in tasks_list:
            tt, pruned_results = enumeration_results[task]
            dictionnary_precompute[task] = pruned_results
            mode_details_post_compute[task] = {}

            # Sort by load-balancing heuristic
            tt = sorted(
                tt,
                key=lambda x: min(
                    [
                        usage_worker_in_chosen_modes[y]
                        for y in x
                        if y in usage_worker_in_chosen_modes
                    ],
                    default=0,
                ),
            )

            # Truncate to max modes per task
            if limit_number_of_mode_per_task:
                number_of_modes = min(max_number_of_mode, len(tt))
                tt = tt[:number_of_modes]
            else:
                number_of_modes = len(tt)

            logger.debug(
                f"  Task {task}: Selected {number_of_modes} mode(s) "
                f"from {len(enumeration_results[task][0])} compliance-filtered candidates"
            )

            # Store selected modes and update usage counters
            if number_of_modes == 0:
                # Fallback: no worker-type assignment found (e.g. source/sink tasks with
                # duration=0 and no skill requirements).  Insert the first original mode
                # stripped of its skill keys so the resulting RCPSP stays feasible.
                first_original_mode = initial_mode_details[task][
                    min(initial_mode_details[task])
                ]
                fallback_mode = {
                    key: first_original_mode[key]
                    for key in first_original_mode
                    if key not in self.multiskill_model.skills_set
                }
                mode_details_post_compute[task][1] = fallback_mode
                logger.debug(
                    f"  Task {task}: No valid modes found — using fallback mode {fallback_mode}"
                )
            else:
                for i in range(number_of_modes):
                    mode_details_post_compute[task][i + 1] = tt[i]
                    for yy in tt[i]:
                        if yy in usage_worker_in_chosen_modes:
                            usage_worker_in_chosen_modes[yy] += 1
        
        if logger.isEnabledFor(logging.DEBUG):
            # Log timing statistics
            if task_times:
                avg_task_time = np.mean(list(task_times.values()))
                max_task = max(task_times, key=task_times.get)
                max_task_time = task_times[max_task]
                logger.debug(f"  Task timing: avg={avg_task_time:.3f}s, max={max_task_time:.3f}s (task {max_task})")
            
            # Log worker type usage statistics
            logger.info("Worker type usage in selected modes:")
            for wt in sorted(usage_worker_in_chosen_modes.keys()):
                usage = usage_worker_in_chosen_modes[wt]
                logger.info(f"  {wt}: {usage} usage(s)")

        stage2_time = time.time() - start - stage1_time
        
        # Stage 3: Build final RCPSP problem
        logger.info(f"{time.time() - start:.2f}s: Constructing final RCPSP problem with selected modes...")
        
        rcpsp_problem = RcpspProblem(
            resources=resources_dict,
            non_renewable_resources=list(self.multiskill_model.non_renewable_resources),
            mode_details=mode_details_post_compute,
            successors=self.multiskill_model.successors,
            horizon=self.multiskill_model.horizon,
            horizon_multiplier=self.multiskill_model.horizon_multiplier,
            tasks_list=self.multiskill_model.tasks_list,
            source_task=self.multiskill_model.source_task,
            sink_task=self.multiskill_model.sink_task,
        )
        
        stage3_time = time.time() - start - stage1_time - stage2_time
        
        # Summary
        total_time = time.time() - start
        logger.info("=" * 80)
        logger.info("CONSTRUCTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.debug(f"  Stage 1 (Clustering): {stage1_time:.2f}s ({100*stage1_time/total_time:.1f}%)")
        logger.debug(f"  Stage 2 (Mode Enumeration): {stage2_time:.2f}s ({100*stage2_time/total_time:.1f}%)")
        logger.debug(f"  Stage 3 (RCPSP Creation): {stage3_time:.2f}s ({100*stage3_time/total_time:.1f}%)")
        logger.info(f"Created RCPSP with {len(rcpsp_problem.tasks_list)} tasks, "
                    f"{len(rcpsp_problem.resources)} resources")
        logger.info(f"Resources: {list(rcpsp_problem.resources.keys())}")
        
        # Log mode statistics
        total_modes = sum(len(mode_details_post_compute[t]) for t in mode_details_post_compute)
        avg_modes_per_task = total_modes / len(rcpsp_problem.tasks_list) if rcpsp_problem.tasks_list else 0
        logger.info(f"Modes: {total_modes} total, {avg_modes_per_task:.2f} avg per task")
        logger.info("=" * 80)
        
        return rcpsp_problem
