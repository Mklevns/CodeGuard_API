"""
Real-time Performance Heatmap for ML Model Diagnostics.
Provides visual performance metrics and bottleneck identification for machine learning models.
"""

import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from pydantic import BaseModel
import psutil
import threading


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: str
    metric_name: str
    value: float
    category: str  # 'cpu', 'memory', 'gpu', 'model', 'training'
    source: str    # 'system', 'pytorch', 'tensorflow', 'custom'
    metadata: Dict[str, Any]


class HeatmapConfig(BaseModel):
    """Configuration for performance heatmap generation."""
    update_interval: float = 1.0  # seconds
    history_window: int = 300     # number of data points to keep
    categories: List[str] = ['cpu', 'memory', 'gpu', 'model', 'training']
    auto_detect_frameworks: bool = True
    include_system_metrics: bool = True
    alert_thresholds: Dict[str, float] = {
        'cpu_usage': 90.0,
        'memory_usage': 85.0,
        'gpu_usage': 95.0,
        'training_loss': 0.01  # rate of change threshold
    }


class MLPerformanceCollector:
    """Collects real-time performance metrics from ML training processes."""
    
    def __init__(self, config: HeatmapConfig):
        self.config = config
        self.metrics_buffer = deque(maxlen=config.history_window)
        self.active_frameworks = set()
        self.collection_active = False
        self.collection_thread = None
        self.custom_metrics = {}
        
    def start_collection(self):
        """Start real-time metric collection."""
        if self.collection_active:
            return
            
        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
    def stop_collection(self):
        """Stop metric collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
    
    def _collection_loop(self):
        """Main collection loop running in separate thread."""
        while self.collection_active:
            try:
                # Collect system metrics
                if self.config.include_system_metrics:
                    self._collect_system_metrics()
                
                # Detect and collect framework-specific metrics
                if self.config.auto_detect_frameworks:
                    self._detect_frameworks()
                    self._collect_framework_metrics()
                
                # Collect custom metrics
                self._collect_custom_metrics()
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Error in metric collection: {e}")
                time.sleep(self.config.update_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        timestamp = datetime.now().isoformat()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_name='cpu_usage',
            value=cpu_percent,
            category='cpu',
            source='system',
            metadata={'cores': psutil.cpu_count()}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_name='memory_usage',
            value=memory.percent,
            category='memory',
            source='system',
            metadata={
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3)
            }
        ))
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self._add_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_name='disk_read_rate',
                value=disk_io.read_bytes / (1024**2),  # MB/s approximation
                category='disk',
                source='system',
                metadata={'read_count': disk_io.read_count}
            ))
    
    def _detect_frameworks(self):
        """Detect active ML frameworks."""
        try:
            # Check for PyTorch processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'torch' in cmdline.lower() or 'pytorch' in cmdline.lower():
                        self.active_frameworks.add('pytorch')
                    if 'tensorflow' in cmdline.lower() or 'tf.' in cmdline.lower():
                        self.active_frameworks.add('tensorflow')
                    if 'gym' in cmdline.lower() or 'stable_baselines' in cmdline.lower():
                        self.active_frameworks.add('reinforcement_learning')
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error detecting frameworks: {e}")
    
    def _collect_framework_metrics(self):
        """Collect framework-specific metrics."""
        timestamp = datetime.now().isoformat()
        
        # PyTorch metrics (if available)
        if 'pytorch' in self.active_frameworks:
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                        gpu_cached = torch.cuda.memory_reserved(i) / (1024**3)   # GB
                        
                        self._add_metric(PerformanceMetric(
                            timestamp=timestamp,
                            metric_name=f'gpu_{i}_memory_allocated',
                            value=gpu_memory,
                            category='gpu',
                            source='pytorch',
                            metadata={'device': i, 'cached_gb': gpu_cached}
                        ))
            except ImportError:
                pass
        
        # TensorFlow metrics (if available)
        if 'tensorflow' in self.active_frameworks:
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for i, gpu in enumerate(gpus):
                    # TensorFlow GPU metrics would go here
                    # This is a placeholder for actual TF GPU monitoring
                    pass
            except ImportError:
                pass
    
    def _collect_custom_metrics(self):
        """Collect custom metrics registered by user code."""
        timestamp = datetime.now().isoformat()
        
        for metric_name, metric_data in self.custom_metrics.items():
            if callable(metric_data):
                try:
                    value = metric_data()
                    self._add_metric(PerformanceMetric(
                        timestamp=timestamp,
                        metric_name=metric_name,
                        value=float(value),
                        category='custom',
                        source='user',
                        metadata={}
                    ))
                except Exception as e:
                    print(f"Error collecting custom metric {metric_name}: {e}")
    
    def _add_metric(self, metric: PerformanceMetric):
        """Add a metric to the buffer."""
        self.metrics_buffer.append(metric)
    
    def register_custom_metric(self, name: str, collector_func: callable):
        """Register a custom metric collector function."""
        self.custom_metrics[name] = collector_func
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_metrics = []
        for metric in self.metrics_buffer:
            metric_time = datetime.fromisoformat(metric.timestamp)
            if metric_time >= cutoff_time:
                recent_metrics.append(metric)
        
        return recent_metrics


class HeatmapGenerator:
    """Generates heatmap visualization data from performance metrics."""
    
    def __init__(self):
        pass
    
    def generate_heatmap_data(self, metrics: List[PerformanceMetric], 
                            grid_size: tuple = (20, 10)) -> Dict[str, Any]:
        """Generate heatmap data structure for visualization."""
        
        # Group metrics by category and time windows
        time_buckets = self._create_time_buckets(metrics, grid_size[0])
        category_data = self._group_by_category(metrics)
        
        # Create heatmap matrix
        heatmap_matrix = []
        categories = list(category_data.keys())
        
        for category in categories:
            category_row = []
            for time_bucket in time_buckets:
                # Calculate average value for this category in this time bucket
                bucket_metrics = [m for m in category_data[category] 
                                if m.timestamp in time_bucket['timestamps']]
                
                if bucket_metrics:
                    avg_value = sum(m.value for m in bucket_metrics) / len(bucket_metrics)
                    # Normalize to 0-1 scale for heatmap
                    normalized_value = self._normalize_value(avg_value, category)
                else:
                    normalized_value = 0.0
                
                category_row.append({
                    'value': normalized_value,
                    'raw_value': avg_value if bucket_metrics else 0,
                    'count': len(bucket_metrics),
                    'timestamp': time_bucket['start_time']
                })
            
            heatmap_matrix.append(category_row)
        
        return {
            'matrix': heatmap_matrix,
            'categories': categories,
            'time_buckets': [tb['start_time'] for tb in time_buckets],
            'grid_size': grid_size,
            'generated_at': datetime.now().isoformat(),
            'total_metrics': len(metrics)
        }
    
    def _create_time_buckets(self, metrics: List[PerformanceMetric], 
                           num_buckets: int) -> List[Dict]:
        """Create time buckets for heatmap columns."""
        if not metrics:
            return []
        
        # Get time range
        timestamps = [datetime.fromisoformat(m.timestamp) for m in metrics]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        # Create equal-sized time buckets
        time_delta = (end_time - start_time) / num_buckets
        buckets = []
        
        for i in range(num_buckets):
            bucket_start = start_time + (time_delta * i)
            bucket_end = start_time + (time_delta * (i + 1))
            
            bucket_timestamps = [
                m.timestamp for m in metrics
                if bucket_start <= datetime.fromisoformat(m.timestamp) < bucket_end
            ]
            
            buckets.append({
                'start_time': bucket_start.isoformat(),
                'end_time': bucket_end.isoformat(),
                'timestamps': bucket_timestamps
            })
        
        return buckets
    
    def _group_by_category(self, metrics: List[PerformanceMetric]) -> Dict[str, List]:
        """Group metrics by category."""
        category_groups = defaultdict(list)
        for metric in metrics:
            category_groups[metric.category].append(metric)
        return dict(category_groups)
    
    def _normalize_value(self, value: float, category: str) -> float:
        """Normalize values to 0-1 scale based on category."""
        # Category-specific normalization ranges
        normalization_ranges = {
            'cpu': (0, 100),      # CPU percentage
            'memory': (0, 100),   # Memory percentage
            'gpu': (0, 100),      # GPU usage percentage
            'disk': (0, 1000),    # MB/s
            'custom': (0, 1),     # Assume custom metrics are pre-normalized
            'model': (0, 1),      # Model-specific metrics
            'training': (0, 10)   # Training metrics (loss, accuracy, etc.)
        }
        
        min_val, max_val = normalization_ranges.get(category, (0, 1))
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))


class PerformanceHeatmapAPI:
    """API endpoints for real-time performance heatmap."""
    
    def __init__(self):
        self.collector = None
        self.generator = HeatmapGenerator()
        self.config = HeatmapConfig()
    
    def start_monitoring(self, config: Optional[HeatmapConfig] = None) -> Dict[str, Any]:
        """Start performance monitoring."""
        if config:
            self.config = config
        
        if self.collector and self.collector.collection_active:
            return {'status': 'already_running', 'message': 'Monitoring already active'}
        
        self.collector = MLPerformanceCollector(self.config)
        self.collector.start_collection()
        
        return {
            'status': 'started',
            'config': asdict(self.config),
            'message': 'Performance monitoring started'
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop performance monitoring."""
        if self.collector:
            self.collector.stop_collection()
            return {'status': 'stopped', 'message': 'Performance monitoring stopped'}
        
        return {'status': 'not_running', 'message': 'No active monitoring'}
    
    def get_heatmap_data(self, minutes: int = 5, 
                        grid_size: tuple = (20, 10)) -> Dict[str, Any]:
        """Get current heatmap data."""
        if not self.collector:
            return {'error': 'Monitoring not started'}
        
        recent_metrics = self.collector.get_recent_metrics(minutes)
        if not recent_metrics:
            return {
                'matrix': [],
                'categories': [],
                'time_buckets': [],
                'message': 'No metrics available yet'
            }
        
        heatmap_data = self.generator.generate_heatmap_data(recent_metrics, grid_size)
        
        # Add performance insights
        heatmap_data['insights'] = self._generate_insights(recent_metrics)
        heatmap_data['alerts'] = self._check_alerts(recent_metrics)
        
        return heatmap_data
    
    def register_custom_metric(self, name: str, collector_func: callable) -> Dict[str, Any]:
        """Register a custom metric collector."""
        if not self.collector:
            return {'error': 'Monitoring not started'}
        
        self.collector.register_custom_metric(name, collector_func)
        return {'status': 'registered', 'metric_name': name}
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        if not self.collector:
            return {'status': 'stopped', 'metrics_count': 0}
        
        return {
            'status': 'running' if self.collector.collection_active else 'stopped',
            'metrics_count': len(self.collector.metrics_buffer),
            'active_frameworks': list(self.collector.active_frameworks),
            'custom_metrics': list(self.collector.custom_metrics.keys())
        }
    
    def _generate_insights(self, metrics: List[PerformanceMetric]) -> List[Dict[str, str]]:
        """Generate performance insights from metrics."""
        insights = []
        
        # Group by category for analysis
        category_groups = defaultdict(list)
        for metric in metrics:
            category_groups[metric.category].append(metric.value)
        
        for category, values in category_groups.items():
            if not values:
                continue
            
            avg_value = sum(values) / len(values)
            max_value = max(values)
            
            if category in ['cpu', 'memory'] and avg_value > 80:
                insights.append({
                    'type': 'warning',
                    'category': category,
                    'message': f'High {category} usage detected (avg: {avg_value:.1f}%)'
                })
            
            if category == 'gpu' and max_value > 95:
                insights.append({
                    'type': 'critical',
                    'category': category,
                    'message': f'GPU usage at maximum capacity ({max_value:.1f}%)'
                })
        
        return insights
    
    def _check_alerts(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        
        # Check against configured thresholds
        for metric in metrics[-10:]:  # Check last 10 metrics
            threshold_key = metric.metric_name
            if threshold_key in self.config.alert_thresholds:
                threshold = self.config.alert_thresholds[threshold_key]
                
                if metric.value > threshold:
                    alerts.append({
                        'timestamp': metric.timestamp,
                        'metric': metric.metric_name,
                        'value': metric.value,
                        'threshold': threshold,
                        'severity': 'high' if metric.value > threshold * 1.1 else 'medium'
                    })
        
        return alerts


# Global API instance
heatmap_api = PerformanceHeatmapAPI()