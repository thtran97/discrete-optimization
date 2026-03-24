import os
import time
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspProblem, VariantMultiskillRcpspProblem
from discrete_optimization.rcpsp_multiskill.parser_imopse import get_data_available, parse_file
from discrete_optimization.rcpsp_multiskill.solvers.multimode_transposition import MultimodeTranspositionMultiskillRcpspSolver
from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import CpMultiskillRcpspSolver, CpSolverName
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import CpSatMultiskillRcpspSolver
from discrete_optimization.rcpsp_multiskill.solvers.lp import MathOptMultiskillRcpspSolver
from discrete_optimization.rcpsp_multiskill.solvers.optal import OptalMSRcpspSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp_multiskill.multiskill_to_rcpsp import MultiSkillToRcpsp
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
    MultimodeConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstanceAnalyzer:
    """Compute structural metrics for a single MS-RCPSP instance."""
    
    @staticmethod
    def analyze(problem: MultiskillRcpspProblem) -> Dict[str, Any]:
        logger.info(f"Analyzing problem: {len(problem.tasks_list)} tasks, {len(problem.employees)} employees")
        
        # Worker-type clustering
        # Check how many distinct worker types there are based on their skill sets
        # This gives an idea of the diversity of the workforce in terms of skills
        from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import PrecomputeEmployeesForTasks
        solver = PrecomputeEmployeesForTasks(ms_rcpsp_problem=problem)
        
        worker_types = solver.skills_dict # list of lists of skills, each inner list is a worker type
        n_worker_types = len(worker_types) # number of worker types
        n_employees = len(problem.employees) # number of employees
        
        # Skill redundancy = sum(number of employees with skill i) / number of skills
        # this is a measure of how many employees have each skill on average
        # the higher it is, the more redundant the skill coverage is 
        # the lower it is, the more specialized the skill coverage is
        skill_counts = {}
        for wt_skills in worker_types:
            for skill in wt_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        avg_skill_redundancy = np.mean(list(skill_counts.values())) if skill_counts else 0
        
        # sum of durations of all tasks, e.g. the trivial lower bound on the makespan
        durations = [problem.mode_details[t][1].get("duration", 0) for t in problem.tasks_list]
        total_work = sum(durations) 
        
        # Analyze "mode" exploration 
        # How many modes would the raw RCPSP have if we construct it without limiting the number of modes per task?
        # MER = total number of modes in the raw RCPSP / number of tasks 
        # raw MER gives an idea of how complex the mode space is for the instance
        # the higher the MER, the more complex the mode space is, and potentially the harder it is for solvers to find good solutions (?)
        algo = MultiSkillToRcpsp(problem)
        start_time = time.time()
        try:
            # Only compute raw modes if the problem is not too large
            if len(problem.tasks_list) <= 300:
                # construct the raw RCPSP without limiting the number of modes per task
                rcpsp_raw = algo.construct_rcpsp_by_worker_type(limit_number_of_mode_per_task=False)
                # calculate total number of modes in the raw RCPSP
                total_raw_modes = sum(len(rcpsp_raw.mode_details[t]) for t in rcpsp_raw.tasks_list) 
                # calculate raw MER
                raw_MER = total_raw_modes / len(problem.tasks_list)
            else:
                total_raw_modes = -1
                raw_MER = -1
        except Exception as e:
            logger.warning(f"Could not compute raw MER: {e}")
            raw_MER = -1
            total_raw_modes = -1
            
        analysis_time = time.time() - start_time
        
        return {
            "n_tasks": len(problem.tasks_list),
            "n_employees": n_employees,
            "n_worker_types": n_worker_types,
            "worker_type_diversity": n_worker_types / n_employees if n_employees > 0 else 0,
            "avg_skill_redundancy": avg_skill_redundancy,
            "total_work": total_work,
            "raw_total_modes": total_raw_modes, # total number of modes in the raw RCPSP
            "raw_MER": raw_MER, # mode exploration ratio = total_raw_modes / n_tasks
            "analysis_time": analysis_time
        }

class BenchmarkRunner:
    """Run MM transposition solver vs Baseline; collect results."""
    
    def __init__(self, time_limit_sec: int = 60):
        self.time_limit = time_limit_sec

    def run_ms_to_mm(self, problem: MultiskillRcpspProblem, strategy: str = "hard") -> Dict[str, Any]:
        logger.info(f"Running MS-to-MM Relaxation ({strategy}) on instance...")
        solver = MultimodeTranspositionMultiskillRcpspSolver(
            problem=problem,
            reconstruction_strategy=strategy,
            reconstruction_cp_time_limit=self.time_limit // 2,
            bounded_slack_delta=10
        )
        
        parameters_cp = ParametersCp.default()
        start_time = time.time()
        result_store = solver.solve(time_limit=self.time_limit // 2, parameters_cp=parameters_cp)
        solve_time = time.time() - start_time
        
        best_sol, fit = result_store.get_best_solution_fit()
        
        if best_sol is not None:
            makespan = best_sol.get_end_time(problem.sink_task)
            feasible = problem.satisfy(best_sol)
        else:
            makespan = -1
            feasible = False
            
        return {
            f"mm_{strategy}_makespan": makespan,
            f"mm_{strategy}_feasible": feasible,
            f"mm_{strategy}_solve_time": solve_time,
            f"mm_{strategy}_fit": fit
        }

    def run_other_solver(self, problem: MultiskillRcpspProblem, solver_class, name: str, init_kwargs=None) -> Dict[str, Any]:
        logger.info(f"Running {name} on instance...")
        try:
            if init_kwargs is None:
                init_kwargs = {}
            if solver_class == CpMultiskillRcpspSolver:
                solver = solver_class(problem=problem, cp_solver_name=CpSolverName.CHUFFED)
            else:
                solver = solver_class(problem=problem)
            
            try:
                solver.init_model(**init_kwargs)
            except Exception as e:
                logger.warning(f"{name} init_model failed or not present: {e}")

            start_time = time.time()
            if solver_class == CpMultiskillRcpspSolver:
                result_store = solver.solve(time_limit=self.time_limit)
            else:
                try:
                    result_store = solver.solve(time_limit=self.time_limit)
                except Exception:
                    parameters_cp = ParametersCp.default()
                    parameters_cp.time_limit = self.time_limit
                    result_store = solver.solve(parameters_cp=parameters_cp, time_limit=self.time_limit)
            solve_time = time.time() - start_time
            
            if result_store is None or not hasattr(result_store, 'get_best_solution_fit'):
                return {f"{name}_makespan": -1, f"{name}_feasible": False, f"{name}_solve_time": solve_time, f"{name}_fit": -1}
                
            best_sol, fit = result_store.get_best_solution_fit()
            
            if best_sol is not None:
                makespan = best_sol.get_end_time(problem.sink_task)
                feasible = problem.satisfy(best_sol)
                if not feasible and getattr(best_sol, "_external_feasible", False):
                    feasible = True
            else:
                makespan = -1
                feasible = False
                
            return {
                f"{name}_makespan": makespan,
                f"{name}_feasible": feasible,
                f"{name}_solve_time": solve_time,
                f"{name}_fit": fit
            }
        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            return {
                f"{name}_makespan": -1,
                f"{name}_feasible": False,
                f"{name}_solve_time": -1,
                f"{name}_fit": -1
            }

    def run_lns(self, problem: MultiskillRcpspProblem) -> Dict[str, Any]:
        logger.info(f"Running LNS on instance...")
        try:
            subsolver = CpSatMultiskillRcpspSolver(problem=problem)
            extractors: list[BaseConstraintExtractor] = [
                SchedulingConstraintExtractor(
                    minus_delta_primary=100,
                    plus_delta_primary=100,
                    minus_delta_secondary=10,
                    plus_delta_secondary=10,
                ),
                MultimodeConstraintExtractor(),
                SubresourcesAllocationConstraintExtractor(),
            ]
            constraints_extractor = ConstraintExtractorList(extractors=extractors)
            constraint_handler = TasksConstraintHandler(
                problem=problem,
                constraints_extractor=constraints_extractor,
            )
            solver = LnsOrtoolsCpSat(
                problem=problem,
                subsolver=subsolver,
                constraint_handler=constraint_handler,
            )
            
            start_time = time.time()
            result_store = solver.solve(
                nb_iteration_lns=100,
                time_limit_subsolver_iter0=min(2, self.time_limit // 5),
                time_limit_subsolver=min(5, self.time_limit // 2),
                time_limit=self.time_limit,
                skip_initial_solution_provider=True,
            )
            solve_time = time.time() - start_time
            
            best_sol, fit = result_store.get_best_solution_fit()
            if best_sol is not None:
                makespan = best_sol.get_end_time(problem.sink_task)
                feasible = problem.satisfy(best_sol)
            else:
                makespan = -1
                feasible = False
                fit = -1
                
            return {
                "lns_makespan": makespan,
                "lns_feasible": feasible,
                "lns_solve_time": solve_time,
                "lns_fit": fit
            }
        except Exception as e:
            logger.error(f"Error running LNS: {e}")
            return {
                "lns_makespan": -1,
                "lns_feasible": False,
                "lns_solve_time": -1,
                "lns_fit": -1
            }

import argparse

def main():
    SOLVER_CHOICES = ["mm_hard", "mm_slack", "mm_soft", "cp", "cpsat", "lns", "optal"]
    parser = argparse.ArgumentParser(description="Benchmark MS-RCPSP relaxation and solvers.")
    parser.add_argument("--solvers", nargs="+", 
                        choices=SOLVER_CHOICES + ["all"],
                        default=["all"],
                        help="Solvers/strategies to run.")
    parser.add_argument("--time_limit", type=int, default=60, help="Time limit per solver in seconds.")
    parser.add_argument("--analyze", action="store_true", help="Analyze instances before solving.")
    args = parser.parse_args()

    if "all" in args.solvers:
        args.solvers = SOLVER_CHOICES

    all_files = get_data_available()
    # Decomment to select a few instances of different sizes for benchmarking
    target_instances = [
        f for f in all_files
        # if "100_5_22_15.def" in f 
        #    or "100_5_64_9.def" in f
        #    or "200_10_84_9.def" in f
        #    or "200_10_25_10.def" in f
        #    or "500_20_24_11.def" in f
    ]
    
    runner = BenchmarkRunner(time_limit_sec=args.time_limit)
    
    # Dictionary to store results per solver
    solver_results = {s: [] for s in args.solvers}
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, file_path in enumerate(target_instances):
        instance_name = os.path.basename(file_path)
        logger.info(f"PROCESS {i+1}/{len(target_instances)}: {instance_name} =============== ")
        
        try:
            problem, _ = parse_file(file_path)
            
            # Phase 1: Analyze the instance (can be skipped without adding argument --analyze)
            if args.analyze:
                metrics = InstanceAnalyzer.analyze(problem)
            else:
                metrics = {}
            metrics["instance"] = instance_name
            
            # Phase 2: Solve with selected solvers/strategies
            if "mm_hard" in args.solvers:
                res = runner.run_ms_to_mm(problem, strategy="hard")
                solver_results["mm_hard"].append({**metrics, **res})
                pd.DataFrame(solver_results["mm_hard"]).to_csv(os.path.join(output_dir, "benchmark_mm_hard.csv"), index=False)

            if "mm_slack" in args.solvers:
                res = runner.run_ms_to_mm(problem, strategy="bounded_slack")
                solver_results["mm_slack"].append({**metrics, **res})
                pd.DataFrame(solver_results["mm_slack"]).to_csv(os.path.join(output_dir, "benchmark_mm_slack.csv"), index=False)

            if "mm_soft" in args.solvers:
                res = runner.run_ms_to_mm(problem, strategy="soft_penalty")
                solver_results["mm_soft"].append({**metrics, **res})
                pd.DataFrame(solver_results["mm_soft"]).to_csv(os.path.join(output_dir, "benchmark_mm_soft.csv"), index=False)

            if "cp" in args.solvers:
                res = runner.run_other_solver(problem, CpMultiskillRcpspSolver, "cp", {"one_ressource_per_task": True})
                solver_results["cp"].append({**metrics, **res})
                pd.DataFrame(solver_results["cp"]).to_csv(os.path.join(output_dir, "benchmark_cp.csv"), index=False)

            if "cpsat" in args.solvers:
                res = runner.run_other_solver(problem, CpSatMultiskillRcpspSolver, "cpsat")
                solver_results["cpsat"].append({**metrics, **res})
                pd.DataFrame(solver_results["cpsat"]).to_csv(os.path.join(output_dir, "benchmark_cpsat.csv"), index=False)

            # if "mathopt" in args.solvers: # TODO: currently not working well, needs review
            #     res = runner.run_other_solver(problem, MathOptMultiskillRcpspSolver, "mathopt")
            #     solver_results["mathopt"].append({**metrics, **res})
            #     pd.DataFrame(solver_results["mathopt"]).to_csv(os.path.join(output_dir, "benchmark_mathopt.csv"), index=False)

            if "optal" in args.solvers:
                res = runner.run_other_solver(problem, OptalMSRcpspSolver, "optal")
                solver_results["optal"].append({**metrics, **res})
                pd.DataFrame(solver_results["optal"]).to_csv(os.path.join(output_dir, "benchmark_optal.csv"), index=False)

            if "lns" in args.solvers:
                res = runner.run_lns(problem)
                solver_results["lns"].append({**metrics, **res})
                pd.DataFrame(solver_results["lns"]).to_csv(os.path.join(output_dir, "benchmark_lns.csv"), index=False)
            
        except Exception as e:
            logger.error(f"Error processing {instance_name}: {e}")

    logger.info("Benchmark complete. Results saved to separate CSV files.")

if __name__ == "__main__":
    main()
