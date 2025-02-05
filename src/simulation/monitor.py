"""
Performance monitoring and process management utilities.
"""

import os
import psutil
from typing import Dict, List, Optional
import time

class ProcessMonitor:
    def __init__(self):
        self.processes: Dict[str, psutil.Process] = {}
        self.start_times: Dict[str, float] = {}
        self.cpu_samples: Dict[str, List[float]] = {}
        self.memory_samples: Dict[str, List[float]] = {}
        self.sample_window = 60  # Keep last 60 samples
        
    def register_process(self, name: str, pid: int):
        """Register a process for monitoring."""
        try:
            process = psutil.Process(pid)
            self.processes[name] = process
            self.start_times[name] = time.time()
            self.cpu_samples[name] = []
            self.memory_samples[name] = []
            print(f"Registered process '{name}' (PID: {pid})")
        except psutil.NoSuchProcess:
            print(f"Warning: Process {pid} not found")
            
    def update(self):
        """Update performance metrics for all registered processes."""
        for name, process in self.processes.items():
            try:
                # Get CPU and memory usage
                cpu_percent = process.cpu_percent()
                memory_percent = process.memory_percent()
                
                # Update samples
                self.cpu_samples[name].append(cpu_percent)
                self.memory_samples[name].append(memory_percent)
                
                # Keep only recent samples
                if len(self.cpu_samples[name]) > self.sample_window:
                    self.cpu_samples[name].pop(0)
                if len(self.memory_samples[name]) > self.sample_window:
                    self.memory_samples[name].pop(0)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"Warning: Lost access to process '{name}'")
                
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics for all processes."""
        stats = {}
        for name in self.processes:
            stats[name] = {
                'cpu_avg': sum(self.cpu_samples[name]) / len(self.cpu_samples[name]) if self.cpu_samples[name] else 0,
                'mem_avg': sum(self.memory_samples[name]) / len(self.memory_samples[name]) if self.memory_samples[name] else 0,
                'uptime': time.time() - self.start_times[name]
            }
        return stats
        
    def print_stats(self):
        """Print current performance statistics."""
        stats = self.get_stats()
        print("\nProcess Statistics:")
        print("-" * 60)
        print(f"{'Process':<15} {'CPU %':>10} {'Memory %':>10} {'Uptime (s)':>10}")
        print("-" * 60)
        for name, metrics in stats.items():
            print(f"{name:<15} {metrics['cpu_avg']:>10.1f} {metrics['mem_avg']:>10.1f} {metrics['uptime']:>10.0f}")
            
    def set_process_priority(self, name: str, nice_value: int = 10):
        """Set process priority (nice value)."""
        if name in self.processes:
            try:
                self.processes[name].nice(nice_value)
                print(f"Set priority for '{name}' to {nice_value}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"Warning: Could not set priority for '{name}'")
                
    def set_process_affinity(self, name: str, cpu_cores: set):
        """Set process CPU affinity."""
        if name in self.processes:
            try:
                os.sched_setaffinity(self.processes[name].pid, cpu_cores)
                print(f"Set CPU affinity for '{name}' to cores {cpu_cores}")
            except (AttributeError, ProcessLookupError):
                print(f"Warning: Could not set CPU affinity for '{name}'")
                
    def optimize_processes(self):
        """Automatically optimize process settings based on system resources."""
        cpu_count = os.cpu_count() or 1
        processes = list(self.processes.keys())
        
        if len(processes) <= cpu_count:
            # We have enough cores to dedicate one to each process
            for i, name in enumerate(processes):
                self.set_process_affinity(name, {i % cpu_count})
                self.set_process_priority(name, 10)
        else:
            # Distribute processes across available cores
            for i, name in enumerate(processes):
                self.set_process_affinity(name, {i % cpu_count})
                self.set_process_priority(name, 10) 